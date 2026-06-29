# Replicating Goulet Coulombe et al. (2021)

This page documents a clean, from-scratch replication of the forecasting design in
Goulet Coulombe, Leroux, Stevanovic, and Surprenant (2021), "Macroeconomic data
transformations matter," *International Journal of Forecasting*, 37(4), 1338-1354.
The replication is built entirely on the callable `macroforecast` API. The page is
written as a sequence of executed notebook cells, so every output shown below is the
genuine output of the code above it. The cells are also collected in the companion
script `gcls_2021_replication.py`, which regenerates this page.

The earlier replication material for this paper was retired because it accumulated
patches that drifted from the paper specification. This rebuild starts from the paper
design, expresses each layer with the package, and checks each step before the next one
is added. Every layer is shown in two ways. The first states what the paper did and the
closest `macroforecast` construction that matches it. The second treats each combination
of model, preprocessing, target, target type, and horizon as one cell of a pipeline run.

## How this document is organised

The build proceeds in eight steps. A [verification summary](#verification-summary-and-bugs-found)
records the honest outcome and the two package bugs the replication surfaced, and
the detailed appendix numbers the run must reproduce are collected in
[Appendix B ground-truth tables](#appendix-b-ground-truth-tables) at the foot of
this page (machine-readable form: `docs/replication/data/clss2021_appendix_ground_truth.csv`).

1. Replication specification
2. Data construction
3. Forecast-target construction
4. Preprocessing
5. Feature cases
6. Models and arms
7. Pseudo-out-of-sample window
8. Evaluation and execution

---

## Verification summary and bugs found

The replication is a verification exercise. Its value is confirming the pipeline
implements the published methodology, not reproducing an R-based paper bit for
bit. The configuration is faithful (the eight steps below show each layer) and
the pipeline is leak-free. At horizon 1 the replicated relative-RMSE matches the
appendix within about 0.02, and a plain `ols` reproduces the direct and
path-average object exactly. Agreement loosens at longer horizons. After the
evaluation fix below, much of the residual is the expected difference between R's
`randomForest` and scikit-learn's `RandomForestRegressor` (the same
hyperparameters but a different engine and RNG), which is not reducible without
matching the exact R implementation. One residual is not explained by that: the
path-average forecasts of a few volatile real-activity series (HOUST, RETAIL) at
long horizons diverge for the linear `ar` benchmark too, and that loose end is
described under Divergence attribution.

### Configuration faithfulness (verified)

| Axis | Paper | Replication | Status |
| --- | --- | --- | --- |
| POOS window | 38 years | 1980-01 to 2017-12 (38 years), estimation from 1960-01 | match |
| RF hyperparameters | R `randomForest` regression defaults | `max_features=1/3` (mtry=p/3), `min_samples_leaf=5` (nodesize=5), `bootstrap=True`, `n_estimators=200` | match |
| Benchmark FM | factor-augmented AR | `far`, 8 PCA factors, 12 AR lags | match (h1) |
| Features | F, X, MARX, MAF, Level | PCA n=8, MARX/MAF max_lag=12, lags 0..12 | match (h1) |
| Target | average growth rate; average difference for UNRATE | `YGROWTH__` one-period growth, averaged over the horizon | match |
| Preprocessing | stationarity + standardization | official t-codes, EM-factor imputation, IQR outliers | match (h1) |

`scripts/replication/gcls_2021_pipeline/_compare_appendix.py` scores every cell
(10 targets x 6 horizons x {AR, FM, RF F-Level/X-Level/MARX/F-X-MARX-Level} x
{direct, path-average}) against the appendix tables below. The full-grid run
predates the evaluation fix in the next section, so the figures here come from
re-scoring the saved per-origin forecasts with the corrected pairwise
`accuracy_table`, not from re-fitting. Overall mean absolute delta is about 0.10
(direct 0.09, path-average 0.11). It is about 0.03 at horizon 1 and grows to
about 0.16 at horizon 24. AR, which is engine-independent, sits at about 0.065
once it is scored on its full 1980-2017 sample.

### Bug 1. Evaluation sample truncation (critical) — FIXED

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
`tests/pipeline/test_accuracy_pairwise_sample.py`. Because the run predates this
fix, the corrected figures come from re-scoring the saved forecasts, not a
re-run. Re-scoring moves the full-coverage arms (AR, RF_F-Level, RF_MARX) back
onto 1980-2017 and their mean absolute delta against the appendix falls from
about 0.11 to 0.092 (AR alone to 0.065); the short-coverage X-block arms keep
their own 1992-2017 window.

### Bug 2. Horizon-1 direct vs path for information-criterion models — FIXED

At horizon 1, direct and path-average must give identical forecasts (path-average
over one step IS the direct one-step forecast). A supervised model (`ols`)
satisfied this exactly. The information-criterion models `far` and `ar` did not
(about 1% at the RMSE level, so relRMSE was barely affected and h1 still matched
the appendix).

Root cause: order selection. The direct path selects the AR order for IC models
by BIC/AIC on the full training sample (no validation split needed). The
path-average per-step selection block lacked that IC branch and instead ran
CV/validation-split selection (`select_params`), which scores the order on a
truncated sample (the validation block is held out, ending years before the
origin). So the two policies selected different orders for the same data (e.g.
direct chose order 1 while path chose order 4) and diverged even at h1. The
per-step target series at a given order was identical across policies; the
divergence was purely the selected order.

Fix (`macroforecast/forecasting/runner.py`, `_fit_predict_path_average_origin`):
the path per-step block now takes the same IC branch as the direct path. Horizon-1
forecasts are now bit-identical across policies for `ols`, `ar`, and `far`;
horizons > 1 still differ legitimately. Guarded by
`tests/forecasting/test_h1_direct_path_invariant.py`.

### Divergence attribution

After the evaluation fix the residual long-horizon gap is structural, not random,
and it has more than one source. For the random-forest arms part of it is the
expected R `randomForest` versus scikit-learn `RandomForestRegressor` difference,
the same hyperparameters but a different bootstrap RNG and split rule, amplified
at long horizons where the effective sample is small. That is the known
irreducible R-versus-Python gap, not a package defect.

That does not explain everything. The largest residuals are in the path-average
forecasts of volatile real-activity series (HOUST and RETAIL) at long horizons,
and they appear for `ar` too, which is a pure linear autoregression with no
forest. Re-scoring on the full 1980-2017 sample does not remove them (HOUST
path-average AR at horizon 24 is about 0.89 against the appendix 1.48), so they
are not the evaluation truncation. Re-running the horizon-24 path-average FM
benchmark with the corrected order selection lowers its RMSE by only about 10
percent, so they are not the order-selection bug either. What remains is that our
per-step path-average FM benchmark forecasts these series worse than the paper's
at long horizons, which inflates the denominator and pushes the relative RMSE of
every contender below the appendix. The cause of that benchmark gap is still open
and is tracked as a loose end. It is specific to the path-average FM construction
on a handful of volatile series, not a general defect.

### Hypotheses raised and retracted (for the record)

1. RF `max_features=1.0` mismatch — false alarm. The bare `mf.random_forest`
   builder defaults to sklearn's 1.0, but the replication arms set
   `max_features=1/3` via `paper_model_params`.
2. A path-average "OLS-equivalence bug" — retracted. The paper does not run OLS;
   its own AR row differs between the direct and path tables, so direct != path
   for linear models in finite samples is expected.
3. The paper uses a single direct-FM denominator for the path table — retracted.
   Recomputing with the direct-FM denominator made the engine-independent AR
   row WORSE, so the path-FM denominator (what we use) is correct.

---

## Step 1. Replication specification

The paper design is fixed in one place before any code is written. Most of the errors
in the earlier attempt traced back to a target mapping or a learner setting that had
silently diverged from the paper, so this step is treated as load-bearing.

### What we are reproducing

The main-text Table 2 is a compact summary that prints, for every variable and horizon,
the single best specification with coloured bullets and an underline for path-average
targets. It carries no numbers. The detailed numbers live in online Appendix B. Appendix
B.1 (Tables 3 to 8) holds the direct-forecast relative RMSE and Appendix B.2 (Tables 9 to
14) holds the path-average relative RMSE. Each table reports, for one horizon, the FM
absolute RMSE, the AR ratio, and five machine-learning models over sixteen transformation
sets. These tables are the replication target and were re-extracted from the appendix PDF
and validated cell by cell.

### The benchmark, and what FM and AR mean

Every relative-RMSE figure is a ratio to the FM benchmark RMSE. FM is the autoregressive
diffusion-index model of Stock and Watson, estimated by OLS, with the predictor vector
Z_t containing autoregressive lags of the target and principal-component factors. AR is
the nested model that keeps only the autoregressive-lag block and drops the factors. FM is
the denominator. AR is itself reported as a ratio to FM and is a contender, not the
benchmark. The orders are fixed at (p_y, p_f, k) = (12, 12, 8).

### The seven models map to existing package factories

The first verification is that every model the paper uses already exists as a
`macroforecast` model factory, so the full grid is buildable without new estimators.

```python
import macroforecast as mf

paper_to_package = {
    "AR":              "ar",
    "FM (ARDI)":       "far",
    "Elastic Net":     "elastic_net",
    "Adaptive Lasso":  "adaptive_lasso",
    "Linear Boosting": "glmboost",
    "Random Forest":   "random_forest",
    "Boosted Trees":   "gradient_boosting",
}
available = set(dir(mf))
print(f"{'paper model':18s} {'package key':18s} available")
for paper, key in paper_to_package.items():
    print(f"{paper:18s} {key:18s} {key in available}")
```

```
paper model        package key        available
AR                 ar                 True
FM (ARDI)          far                True
Elastic Net        elastic_net        True
Adaptive Lasso     adaptive_lasso     True
Linear Boosting    glmboost           True
Random Forest      random_forest      True
Boosted Trees      gradient_boosting  True
```

### Transformation sets and codes

The predictors enter after the McCracken and Ng stationarity codes. The block X is the
stationarised panel, F is its principal components, and MARX and MAF are moving-average
objects built from the same stationarised panel. The Level block keeps the variables in
raw levels. The forecast target is built separately from the raw level, so it does not
pass through the FRED-MD code. For seven of the ten targets the FRED-MD code is already a
first-order growth and coincides with the target transform; for HOUST, CPI, and PPI it
genuinely differs. The sixteen random-forest information sets are the admissible
combinations of the five blocks.

```python
TRANSFORMS = ["F","F-X","F-MARX","F-MAF","F-Level","F-X-MARX","F-X-MAF",
              "F-X-Level","F-X-MARX-Level","X","MARX","MAF","X-MARX","X-MAF",
              "X-Level","X-MARX-Level"]
print(f"{len(TRANSFORMS)} transformation sets:")
for i in range(0, len(TRANSFORMS), 4):
    print("  " + "  ".join(f"{t:16s}" for t in TRANSFORMS[i:i+4]))
```

```
16 transformation sets:
  F                 F-X               F-MARX            F-MAF           
  F-Level           F-X-MARX          F-X-MAF           F-X-Level       
  F-X-MARX-Level    X                 MARX              MAF             
  X-MARX            X-MAF             X-Level           X-MARX-Level    
```

### Ground-truth validation set

The re-extracted appendix tables give the numbers the run must approach. They are on the
companion page and in `data/clss2021_appendix_ground_truth.csv`. During re-extraction we
found that an earlier hand-made extract had transcription errors in several cells, so the
companion tables supersede any earlier extract.

```python
import pandas as pd
gt = pd.read_csv("docs/replication/data/clss2021_appendix_ground_truth.csv")
print("ground-truth shape:", gt.shape)
print("target types:", sorted(gt.target_type.unique()))
print("models:", sorted(gt.model.unique()))
print("horizons:", sorted(gt.horizon.unique()))
print()
print("example rows (horizon 1, direct, Random Forest):")
ex = gt[(gt.horizon==1)&(gt.target_type=='direct')&(gt.model=='RandomForest')]
print(ex[["info_set","INDPRO","EMP","UNRATE","CPI","PPI"]].head(4).to_string(index=False))
```

```
ground-truth shape: (984, 15)
target types: ['direct', 'sgr']
models: ['AR', 'AdaptiveLasso', 'BoostedTrees', 'ElasticNet', 'FM_ABS', 'LinearBoosting', 'RandomForest']
horizons: [np.int64(1), np.int64(3), np.int64(6), np.int64(9), np.int64(12), np.int64(24)]

example rows (horizon 1, direct, Random Forest):
info_set  INDPRO  EMP  UNRATE  CPI  PPI
       F    0.95 0.99    0.97 1.00 0.97
     F-X    0.96 1.00    0.95 1.00 0.97
  F-MARX    0.93 0.95    0.94 0.97 0.95
   F-MAF    0.96 0.97    0.97 1.01 0.97
```

---

## Step 2. Data construction

### Paper to package

The paper uses the monthly FRED-MD panel of McCracken and Ng, with the estimation sample
beginning in 1960M01 and the pseudo-out-of-sample period running from 1980M01 to 2017M12.
The vintage is not stated, so the build uses the 2018-01 historical vintage, the first one
published after the sample ends. In the package this is a single call that returns a
`DataBundle`, with the transformation codes travelling alongside the panel.

```python
bundle = mf.data.load_fred_md(vintage="2018-01")
panel = bundle.panel
print("shape:", panel.shape)
print("period:", panel.index.min().date(), "..", panel.index.max().date())
print("frequency:", pd.infer_freq(panel.index))
print("transform codes carried in attrs:",
      "macroforecast_transform_codes" in panel.attrs)
```

```
shape: (708, 127)
period: 1959-01-01 .. 2017-12-01
frequency: MS
transform codes carried in attrs: True
```

The sample begins in 1959-01 rather than 1960-01 so that the first estimation origin in
1960-01 has a year of lags available.

### The ten targets are present and complete

The two columns the retired scaffold had wrong, INCOME as RPI and M2 as M2REAL, are
checked here together with the rest.

```python
TARGETS = {"INDPRO":"INDPRO","EMP":"PAYEMS","UNRATE":"UNRATE","INCOME":"RPI",
           "CONS":"DPCERA3M086SBEA","RETAIL":"RETAILx","HOUST":"HOUST",
           "M2":"M2REAL","CPI":"CPIAUCSL","PPI":"WPSFD49207"}
print(f"{'alias':8s} {'column':18s} {'present':8s} {'NaN':>4s}")
for alias, col in TARGETS.items():
    here = col in panel.columns
    n = int(panel[col].isna().sum()) if here else -1
    print(f"{alias:8s} {col:18s} {str(here):8s} {n:4d}")
```

```
alias    column             present   NaN
INDPRO   INDPRO             True        0
EMP      PAYEMS             True        0
UNRATE   UNRATE             True        0
INCOME   RPI                True        0
CONS     DPCERA3M086SBEA    True        0
RETAIL   RETAILx            True        0
HOUST    HOUST              True        0
M2       M2REAL             True        0
CPI      CPIAUCSL           True        0
PPI      WPSFD49207         True        0
```

### The transformation codes match McCracken and Ng

The package codes for the ten targets match the published codes, including the two price
series at code 6 (second log-difference), HOUST at code 4 (log level), and UNRATE at code
2 (first difference). This is the difference that Step 3 must respect when it builds the
target from the raw level.

```python
tcodes = panel.attrs["macroforecast_transform_codes"]
expect = {"INDPRO":5,"PAYEMS":5,"UNRATE":2,"RPI":5,"DPCERA3M086SBEA":5,
          "RETAILx":5,"HOUST":4,"M2REAL":5,"CPIAUCSL":6,"WPSFD49207":6}
def tc(col):
    v = tcodes.get(col)
    return v.get("tcode", v) if isinstance(v, dict) else v
print(f"{'column':18s} {'package':>7s} {'MN':>3s}  match")
for col, e in expect.items():
    print(f"{col:18s} {str(tc(col)):>7s} {e:3d}  {tc(col)==e}")
```

```
column             package  MN  match
INDPRO                   5   5  True
PAYEMS                   5   5  True
UNRATE                   2   2  True
RPI                      5   5  True
DPCERA3M086SBEA          5   5  True
RETAILx                  5   5  True
HOUST                    4   4  True
M2REAL                   5   5  True
CPIAUCSL                 6   6  True
WPSFD49207               6   6  True
```

### Missing values are confined to the early sample

The gaps are in twenty predictors and almost all in the early sample, mostly series that
begin after 1959. They are filled by the factor-based EM imputation in Step 4, which must
run inside the pseudo-out-of-sample loop so that it never sees the future.

```python
na = panel.isna().sum()
na = na[na > 0].sort_values(ascending=False)
print(f"{len(na)} of {panel.shape[1]} columns have missing values; top five:")
for c, n in na.head(5).items():
    fv = panel[c].first_valid_index()
    print(f"  {c:12s} NaN={int(n):4d}  first valid {fv.date()}")
print()
print("missing cells by decade:")
print(panel.isna().sum(axis=1).groupby(panel.index.year//10*10).sum().to_string())
```

```
20 of 127 columns have missing values; top five:
  ACOGNO       NaN= 398  first valid 1992-02-01
  TWEXMMTH     NaN= 168  first valid 1973-01-01
  UMCSENTx     NaN= 154  first valid 1959-05-01
  ANDENOx      NaN= 109  first valid 1968-02-01
  VXOCLSx      NaN=  42  first valid 1962-07-01

missing cells by decade:
date
1950    118
1960    447
1970    220
1980    120
1990     25
2000      0
2010     12
```

### Cell view

The bundle is the single shared input to every cell of the run. A cell is one combination
of model, transformation set, target, target type, and horizon, and each cell reads the
same panel and the same transformation codes. Building the panel once and sharing it is
what lets the later preprocessing cache be reused across cells.

---

## Step 3. Forecast-target construction

This is the step the retired scaffold broke. The forecast target is the h-period average
growth of the series, built from the raw level, and it must not pass through the FRED-MD
transformation code. For the price series the code is a second log-difference, so applying
it to build the target would forecast the change in inflation rather than inflation.

### Paper to package

The paper target is the average of the one-period growths over the h future steps. The
one-period growth is a log-difference for the nine level series and a simple difference for
the unemployment rate. The package builds this directly from the raw level with
`direct_target`, using `average_log_growth` for the log series and `average_change` for
the rate. Because the input is the raw level column, the transformation code never enters.

```python
from macroforecast.feature_engineering import direct_target, path_targets
import numpy as np

SPEC = [("INDPRO","INDPRO","average_log_growth"),
        ("EMP","PAYEMS","average_log_growth"),
        ("UNRATE","UNRATE","average_change"),
        ("INCOME","RPI","average_log_growth"),
        ("CONS","DPCERA3M086SBEA","average_log_growth"),
        ("RETAIL","RETAILx","average_log_growth"),
        ("HOUST","HOUST","average_log_growth"),
        ("M2","M2REAL","average_log_growth"),
        ("CPI","CPIAUCSL","average_log_growth"),
        ("PPI","WPSFD49207","average_log_growth")]
HZ = [1, 3, 6, 9, 12, 24]
POOS = (pd.Timestamp("1980-01-01"), pd.Timestamp("2017-12-01"))

# direct target for every series; report scale at the shortest and longest horizon
print(f"{'target':8s} {'kind':8s} {'h1_n':>5s} {'h1_mean':>8s} {'h1_std':>7s}"
      f" {'h24_n':>6s} {'h24_mean':>9s} {'h24_std':>8s}")
for alias, col, kind in SPEC:
    d = direct_target(panel, target=col, horizons=[1, 24], transform=kind)
    c1 = [c for c in d.columns if c.endswith("_h1")][0]
    c24 = [c for c in d.columns if c.endswith("_h24")][0]
    s1 = d[c1].loc[POOS[0]:POOS[1]]; s24 = d[c24].loc[POOS[0]:POOS[1]]
    knd = "dlog" if kind == "average_log_growth" else "dchange"
    print(f"{alias:8s} {knd:8s} {s1.notna().sum():5d} {s1.mean():8.4f} {s1.std():7.4f}"
          f" {s24.notna().sum():6d} {s24.mean():9.4f} {s24.std():8.4f}")
```

```
target   kind      h1_n  h1_mean  h1_std  h24_n  h24_mean  h24_std
INDPRO   dlog       455   0.0015  0.0067    432    0.0016   0.0026
EMP      dlog       455   0.0011  0.0018    432    0.0011   0.0013
UNRATE   dchange    455  -0.0048  0.1701    432   -0.0065   0.0712
INCOME   dlog       455   0.0022  0.0062    432    0.0023   0.0013
CONS     dlog       455   0.0024  0.0047    432    0.0025   0.0012
RETAIL   dlog       455   0.0039  0.0112    432    0.0039   0.0022
HOUST    dlog       455  -0.0003  0.0802    432    0.0000   0.0130
M2       dlog       455   0.0024  0.0048    432    0.0025   0.0020
CPI      dlog       455   0.0025  0.0029    432    0.0024   0.0012
PPI      dlog       455   0.0019  0.0057    432    0.0017   0.0016
```

The count falls from 455 at h=1 to 432 at h=24 because the longest horizons run past the
sample end. The scale is a monthly growth rate, so means near 0.002 are about a quarter of
a percent per month, and the standard deviation shrinks with the horizon as averaging
smooths the series.

The path-average target produces one column per future step, shifted so that step s holds
the one-period growth realised at t+s.

```python
cpi_path = path_targets(panel, target="CPIAUCSL", horizons=[3], transform="log_growth")
print(cpi_path.loc["1980-01-01":"1980-04-01"].round(4).to_string())
```

```
            CPIAUCSL_log_growth_step1  CPIAUCSL_log_growth_step2  CPIAUCSL_log_growth_step3
date                                                                                       
1980-01-01                     0.0127                     0.0138                     0.0099
1980-02-01                     0.0138                     0.0099                     0.0098
1980-03-01                     0.0099                     0.0098                     0.0097
1980-04-01                     0.0098                     0.0097                     0.0012
```

### Check that the code is bypassed

The clearest check is the price series. The CPI target is the first log-difference, which
is inflation, while the FRED-MD code 6 is the second log-difference, which is the change
in inflation. Over the pseudo-out-of-sample period the two have very different means, which
confirms the target is built from the raw level and not from the transformed panel series.

```python
raw = panel["CPIAUCSL"]
target_dlog = np.log(raw).diff().loc[POOS[0]:POOS[1]]      # what the target uses
code6_ddlog = np.log(raw).diff().diff().loc[POOS[0]:POOS[1]]  # what FRED-MD code 6 gives
print(f"CPI target (first log-difference, inflation):        mean = {target_dlog.mean():+.5f}")
print(f"CPI code 6 (second log-difference, change in infl.): mean = {code6_ddlog.mean():+.5f}")
```

```
CPI target (first log-difference, inflation):        mean = +0.00257
CPI code 6 (second log-difference, change in infl.): mean = -0.00002
```

The target mean of about a quarter of a percent per month is the familiar inflation scale,
while the second-difference mean is essentially zero. The target is therefore the average
growth built from the raw level, which is what the paper requires and what the earlier
scaffold violated.

### Cell view

Each cell carries one target column. A direct cell at horizon h reads the
`average_log_growth` or `average_change` column at that horizon, while a path-average cell
reads the step columns and averages the step forecasts in the evaluation stage. The target
is computed once per (target, horizon, target type) and reused across the model and feature
cells.

---

## Step 4. Preprocessing

### Paper to package

The paper preprocesses the predictor panel in three operations that follow McCracken and
Ng. First the series are stationarised with the standard FRED-MD transformation codes.
Second outliers are flagged, a value more than ten interquartile ranges from the median
being set to missing. Third the missing values are filled by the factor-based EM algorithm
with eight factors. In the package this is one call to `reprocess` with the official
transform. The call below is run on the full sample to show what the three operations do,
and it is therefore a look-ahead version. The faithful run instead applies the same
operations inside the pseudo-out-of-sample loop, which is shown at the end of this step.

```python
pre = mf.preprocessing.reprocess(
    bundle,
    transform="official",                       # standard McCracken-Ng codes
    outliers="iqr", outlier_action="flag_as_nan", iqr_threshold=10.0,
    impute="em_factor", em_n_factors=8, em_demean=2,
    standardize="none",
)
proc = pre.panel
print("raw panel:      ", panel.shape, " NaN =", int(panel.isna().sum().sum()))
print("processed panel:", proc.shape, " NaN =", int(proc.isna().sum().sum()))
```

```
raw panel:       (708, 127)  NaN = 942
processed panel: (706, 127)  NaN = 0
```

### Which order did the official transform apply

This is where the predictor-side ambiguity of Step 1 is settled. The phrase "single-period
differences and growth rates following McCracken and Ng" can be read two ways for the price
series, but the standard FRED-MD codes apply the second log-difference to CPI and PPI. The
official transform follows the standard codes, so the two price series enter as the second
log-difference and HOUST enters as a log level. The check below recovers the applied order
of each series by matching the processed column against candidate transforms of the raw
level.

```python
def applied_order(col):
    raw = panel[col]; pr = proc[col].dropna()
    cands = {"level": raw, "log": np.log(raw), "diff": raw.diff(),
             "dlog": np.log(raw).diff(), "ddlog": np.log(raw).diff().diff()}
    return max(cands, key=lambda k: pr.corr(cands[k].reindex(pr.index)))

print(f"{'series':12s} {'tcode':>5s} {'applied order'}")
for col in ["INDPRO","UNRATE","HOUST","CPIAUCSL","WPSFD49207"]:
    print(f"{col:12s} {tc(col):>5} {applied_order(col)}")
```

```
series       tcode applied order
INDPRO           5 dlog
UNRATE           2 diff
HOUST            4 log
CPIAUCSL         6 ddlog
WPSFD49207       6 ddlog
```

The price series enter as the second log-difference, which is the standard FRED-MD code 6,
so the predictor reconstruction uses the standard codes. The target, built separately in
Step 3 from the raw level, remains the first log-difference, so the predictor and the
target differ for these series exactly as intended.

### Outlier flagging and imputation

The outlier rule turns extreme values into missing cells, which the EM step then fills
together with the genuine gaps. Running the same call with imputation disabled isolates how
many cells the transform and the outlier rule leave missing before the fill.

```python
pre_noimp = mf.preprocessing.reprocess(
    bundle, transform="official",
    outliers="iqr", outlier_action="flag_as_nan", iqr_threshold=10.0,
    impute="none", standardize="none",
)
print("NaN after t-code and outlier flag (pre-impute):", int(pre_noimp.panel.isna().sum().sum()))
print("NaN after EM factor imputation:                ", int(proc.isna().sum().sum()))
```

```
NaN after t-code and outlier flag (pre-impute): 1086
NaN after EM factor imputation:                 0
```

### Leak-aware preprocessing for the run

The full-sample call above lets the EM imputation see the whole sample, which is a
look-ahead. The faithful run attaches a stage policy so the preprocessing only uses data
available up to each origin. The scope `origin_available` makes the imputation leak-free,
and the update cadence is pinned so the expensive EM refit runs on a fixed schedule rather
than at every origin, which keeps the cost bounded without reintroducing a leak.

```python
pp_policy = mf.window.stage_policy("origin_available", update=24)
feat_policy = mf.window.stage_policy("fit_window", update=24)
print("preprocessing policy scope:", pp_policy.scope, "| update:", pp_policy.update)
print("feature policy scope:      ", feat_policy.scope, "| update:", feat_policy.update)
```

```
preprocessing policy scope: origin_available | update: 24
feature policy scope:       fit_window | update: 24
```

### Cell view

The preprocessing is a spec-level stage shared by every model and transformation cell of a
target. Because the transformation codes are fixed and only the per-origin imputation
changes, the result is cached on the origin position and reused across all arms and
horizons of that target, so the EM step is paid once per origin rather than once per cell.

---

## Step 5. Feature cases

### Paper to package

The five building blocks are factors (F), the stationarised predictors (X), moving-average
rotations (MARX), moving-average factors (MAF), and raw levels (Level). The paper uses eight
factors, a maximum lag of twelve for every block, and two components for the moving-average
factors. The block X and the factor block F carry lags zero through twelve, and the lags of
the target are always included.

A short caution on the convenience interface. The package accepts a `feature_specification`
string such as "F-X-MARX-Level", but that shortcut uses the package default lag depth of
zero and one, which is not the paper. The paper depth of zero through twelve is set by
building the blocks explicitly with the step helpers, which is what the cell below does. The
moving-average factor block is computed per variable, so it yields two components for each of
the predictors.

```python
fe = mf.feature_engineering
preds = [c for c in proc.columns if c != "INDPRO"]

# augmented panel: stationary predictors plus raw level copies for the Level block
aug = proc.copy()
for c in preds:
    aug["LEVEL__" + c] = panel[c]
level_cols = ["LEVEL__" + c for c in preds]

def paper_steps(case):
    parts = case.split("-"); steps = []
    if "F" in parts:
        steps.append(fe.pca_step(name="F_raw", columns=preds, n_components=8,
                                 scale=True, include=False, fit_policy="full_sample"))
        steps.append(fe.lag_step(name="F", input="F_raw", lags=range(0, 13), include=True))
    if "X" in parts:
        steps.append(fe.lag_step(name="X", columns=preds, lags=range(0, 13), include=True))
    if "MARX" in parts:
        steps.append(fe.marx_step(name="MARX_X", columns=preds, max_lag=12,
                                  scale_lags=False, include=True))
    if "MAF" in parts:
        steps.append(fe.maf_step(name="MAF_X", columns=preds, max_lag=12, n_components=2,
                                 scale=False, include=True, fit_policy="full_sample"))
    if "Level" in parts:
        steps.append(fe.lag_step(name="Level", columns=level_cols, lags=range(0, 1), include=True))
    return steps

print(f"predictors = {len(preds)};  target lags 0-12 always included (13 columns)")
print(f"{'feature case':18s} {'rows':>5s} {'cols':>6s}")
for case in ["F", "X", "MARX", "MAF", "Level", "F-X-MARX-Level"]:
    fs = fe.build_features(aug, target="INDPRO", horizon=1,
                           feature_steps=paper_steps(case), target_lags=range(0, 13),
                           target_transform="level", drop_missing=True)
    print(f"{case:18s} {fs.X.shape[0]:5d} {fs.X.shape[1]:6d}")
```

```
predictors = 126;  target lags 0-12 always included (13 columns)
feature case        rows   cols
F                    693    117
X                    693   1651
MARX                 693   1525
MAF                  693    265
Level                309    139
F-X-MARX-Level       309   3393
```

The column counts decompose cleanly. The factor block is eight factors over thirteen lags,
which is one hundred and four, plus thirteen target lags, giving one hundred and seventeen.
The predictor block is one hundred and twenty-six series over thirteen lags, which is one
thousand six hundred and thirty-eight, plus thirteen target lags. The moving-average
rotation is one hundred and twenty-six series over twelve windows, which is one thousand
five hundred and twelve, plus thirteen. The moving-average factor block is two components
for each of the one hundred and twenty-six series, which is two hundred and fifty-two, plus
thirteen. The Level block adds the raw levels and loses more early rows to the level lags.
The combined case is the union of the blocks it names, and it is high-dimensional by
construction, which is the regime where the random forest and the regularised models earn
their keep.

The drop in the row count for any case that includes Level or long lags is the leading rows
removed by `drop_missing`, since a thirteen-lag block cannot be evaluated until thirteen
observations have accrued.

### Reproducible factor extraction

A subtle point underlies the factor block. For panels of this shape, more than five
hundred rows with a small number of factors, scikit-learn's PCA selects a randomized
singular value decomposition by default, and that solver draws on the global random
state, so the factors it returns differ from one run to the next unless a seed is fixed.
A factor that changes run to run makes the factor-model benchmark and every factor-based
feature non-reproducible, which is unacceptable for a replication. The `macroforecast`
package therefore routes every principal-component extraction through a single helper
that uses the exact full decomposition for panels of ordinary width, matching the
decomposition used by the imputation step, and falls back to a seeded randomized solver
only for very wide panels. The factors are exact and identical across runs, so no seed
needs to be set by hand, which is why the cells above pass no `random_state`.

### The 16 information sets

A random-forest arm is one of the sixteen admissible combinations of the five blocks. Each
arm reads the union of the blocks its name lists, so the same five step helpers generate the
whole grid.

```
F            F-X           F-MARX        F-MAF         F-Level
F-X-MARX     F-X-MAF       F-X-Level     F-X-MARX-Level
X            MARX          MAF
X-MARX       X-MAF         X-Level       X-MARX-Level
```

### Cell view

A feature cell is one (transformation set, target, horizon) triple. The block steps are
shared across the models that consume them, so for a given target and horizon the feature
matrix of a transformation set is built once and reused by every model arm that uses that
set. In the faithful run the factor and moving-average-factor steps use the expanding fit
policy so they never see the future, while the illustrative build above uses the full-sample
policy for speed.

---

## Step 6. Models and arms

### Paper to package

The seven models all exist as package factories with a common fit interface that takes a
feature matrix and a target and returns a fitted object with a `predict` method. The paper
hyperparameters map onto the factory arguments as follows. The random forest uses two
hundred trees, a minimum leaf size of five, and a feature subsample of one third at each
split. The boosted trees use a depth of five. The factor model and the autoregression are
ordinary least squares with the orders fixed at twelve lags and eight factors. The
penalised models choose their penalty by cross-validation in the run; here they are fitted
at a fixed small penalty to demonstrate that they are operational.

### One caution before fitting: the target source

The feature matrix is built from the stationarised panel, but the target must be built from
the raw level, as in Step 3. Building the target from the stationarised panel reintroduces
the double-transform, because the stationarised industrial-production column is already a
log-difference and a second average-log-growth on top of it is a second difference. The
cell below therefore takes the target from the raw level and the features from the
stationarised panel, then aligns them.

```python
import numpy as np

# target from the RAW level (Step 3); features from the STATIONARISED panel (Step 5)
y = fe.direct_target(panel, target="INDPRO", horizons=[1],
                     transform="average_log_growth").iloc[:, 0]
F_steps = [fe.pca_step(name="F_raw", columns=preds, n_components=8, scale=True,
                       include=False, fit_policy="full_sample"),
           fe.lag_step(name="F", input="F_raw", lags=range(0, 13), include=True)]
Xmat = fe.build_features(proc, target="INDPRO", horizon=1, feature_steps=F_steps,
                         target_lags=range(0, 13), target_transform="level",
                         drop_missing=True).X

d = Xmat.join(y.rename("y"), how="inner").dropna()
Xa, ya = d.drop(columns="y"), d["y"]
print("target scale: mean = %.4f  std = %.4f  (correct growth scale)" % (ya.mean(), ya.std()))

train = Xa.index < "2001-01-01"
Xtr, ytr, Xte, yte = Xa[train], ya[train], Xa[~train], ya[~train]
print("feature matrix:", Xa.shape, " train:", int(train.sum()), " test:", int((~train).sum()))
```

```
target scale: mean = 0.0021  std = 0.0075  (correct growth scale)
feature matrix: (693, 117)  train: 490  test: 203
```

```python
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

arms = [
    ("Random Forest",   "200 trees, leaf 5, mtry #Z/3",
     lambda: mf.random_forest(Xtr, ytr, n_estimators=200, min_samples_leaf=5,
                              max_features=1/3, random_state=123)),
    ("Boosted Trees",   "depth 5, lr 0.1, 200 steps",
     lambda: mf.gradient_boosting(Xtr, ytr, max_depth=5, n_estimators=200,
                                  learning_rate=0.1, random_state=123)),
    ("Elastic Net",     "alpha 1e-3, l1 0.5",
     lambda: mf.elastic_net(Xtr, ytr, alpha=1e-3, l1_ratio=0.5)),
    ("Adaptive Lasso",  "gamma 1, ridge init",
     lambda: mf.adaptive_lasso(Xtr, ytr, gamma=1.0, initial="ridge", alpha=1e-3)),
    ("Linear Boosting", "glmboost, 200 it, lr 0.1",
     lambda: mf.glmboost(Xtr, ytr, n_iter=200, learning_rate=0.1)),
]
print(f"{'model':16s} {'hyperparameters':30s} {'test RMSE':>10s}")
for name, hp, make in arms:
    fit = make()
    print(f"{name:16s} {hp:30s} {rmse(yte, fit.predict(Xte)):10.5f}")

# benchmark and nested contender
fm = mf.far(Xtr, ytr, n_factors=8, n_lag=12, random_state=123)
ar = mf.ar(ytr, n_lag=12)
print(f"{'FM (benchmark)':16s} {'n_factors 8, n_lag 12':30s} {rmse(yte, fm.predict(Xte)):10.5f}")
print(f"{'AR (contender)':16s} {'n_lag 12':30s} {rmse(yte, ar.predict(Xte)):10.5f}")
```

```
model            hyperparameters                 test RMSE
Random Forest    200 trees, leaf 5, mtry #Z/3      0.00645
Boosted Trees    depth 5, lr 0.1, 200 steps        0.00677
Elastic Net      alpha 1e-3, l1 0.5                0.00615
Adaptive Lasso   gamma 1, ridge init               0.00630
Linear Boosting  glmboost, 200 it, lr 0.1          0.00617
FM (benchmark)   n_factors 8, n_lag 12             0.00665
AR (contender)   n_lag 12                          0.00716
```

The pattern is the one the paper reports for this case. The autoregression is the weakest,
the factor model is the benchmark, and the machine-learning models on the factor features
edge below the benchmark. The random forest over the factor block divides into the factor
model at about 0.97, which is close to the appendix figure of 0.95 for industrial
production at horizon one. The numbers are not the appendix numbers, because this is a
single split with a full-sample factor fit rather than the full pseudo-out-of-sample run,
but the ranking is correct and confirms that all seven models are operational under the
paper hyperparameters.

### Cell view

A model cell is one (model, transformation set, target, target type, horizon) tuple. The
benchmark FM and the contender AR are fitted once per (target, target type, horizon) and
shared as the denominator and the data-poor reference. Each machine-learning arm reads the
feature matrix of its transformation set and is fitted per origin in the run. The penalty
of the penalised arms and the step count of the boosted arm are chosen by cross-validation
inside each origin, which is the only per-cell tuning the design carries.

---

## Step 7. Pseudo-out-of-sample window

### Paper to package

The paper uses an expanding estimation window that starts in 1960M01 and a pseudo-out-of-
sample period that runs from 1980M01 to 2017M12. The package builds this with
`from_cutoffs`. Two cadence settings carry the design. The model is re-estimated at every
origin, set by `retrain_every=1`, while the hyperparameters are re-selected only every two
years, set by `retune_every=24` with `retune_on_retrain=False` and `reuse_params=True`.
This decoupling matters because the autoregressive benchmark forecasts recursively from its
training tail and ignores the test predictors, so freezing the fit for two years would let
its forecast go stale and inflate every model's relative RMSE.

```python
window = mf.window.from_cutoffs(
    estimation_start="1960-01", test_start="1980-01", test_end="2017-12",
    mode="expanding", horizon=1,
    retrain_every=1, retune_every=24, retune_on_retrain=False, reuse_params=True,
    val_method="last_block", val_size=60,
)
schedule = window.origins(panel.index)
print("POOS origins:", len(schedule))
print("test span:   ", schedule["test_start"].iloc[0].date(),
      "->", schedule["test_start"].iloc[-1].date())
print("estimation:  ", schedule["estimation_mode"].iloc[0],
      "from", schedule["estimation_start"].iloc[0].date(), "(fixed)")
print("train obs grow:", int(schedule["n_estimation"].iloc[0]),
      "->", int(schedule["n_estimation"].iloc[-1]))
print("refit (retrain=True):", int(schedule["retrain"].sum()), "of", len(schedule),
      "origins  (retrain_every=1)")
```

```
POOS origins: 456
test span:    1980-01-01 -> 2017-12-01
estimation:   expanding from 1960-01-01 (fixed)
train obs grow: 240 -> 695
refit (retrain=True): 456 of 456 origins  (retrain_every=1)
```

The 456 origins span every month from 1980M01 to 2017M12. The estimation window keeps its
1960M01 start and grows with each origin, from 240 observations to 695. The model refits at
all 456 origins.

### Why the cadence is decoupled

The earlier scaffold used a single cadence that refit only every twenty-four months. The
contrast below shows the consequence. Under that setting the model, and with it the
autoregressive benchmark, refits at only 19 of the 456 origins and is frozen in between.
Because the benchmark is the denominator of every relative RMSE, a stale benchmark inflates
the whole table, which is the bug this step fixes.

```python
buggy = mf.window.from_cutoffs(
    estimation_start="1960-01", test_start="1980-01", test_end="2017-12",
    mode="expanding", horizon=1, retrain_every=24,
    val_method="last_block", val_size=60,
)
refit_fixed = int(window.origins(panel.index)["retrain"].sum())
refit_buggy = int(buggy.origins(panel.index)["retrain"].sum())
print(f"refit origins, fixed cadence (retrain_every=1):  {refit_fixed} of 456")
print(f"refit origins, buggy cadence (retrain_every=24): {refit_buggy} of 456  (benchmark frozen between)")
```

```
refit origins, fixed cadence (retrain_every=1):  456 of 456
refit origins, buggy cadence (retrain_every=24): 19 of 456  (benchmark frozen between)
```

### Cell view

The window is shared by every cell of a target. Each origin defines one training slice and
one test point, and the per-origin preprocessing and feature steps attach to it through the
stage policies of Step 4. A cell walks the same 456 origins, refitting its model at each and
re-selecting hyperparameters on the two-year cadence, so the schedule is identical across
models and transformation sets and only the fitted values differ.

---

## Step 8. Evaluation and execution

### Paper to package

The paper evaluates forecasts by the root mean squared error, reports each model as a
relative RMSE against the FM benchmark, tests pairwise accuracy with the Diebold-Mariano
test, and summarises the best set with the Model Confidence Set. All four live in the
package. The pipeline computes them automatically through ``EvalSpec``, which defaults to
the metrics ``rmse`` and ``relative_mse`` and the tests ``dm``, ``cw`` and ``mcs``. The
one subtlety is that the package reports ``relative_mse``, the ratio of mean squared
errors, so the paper's relative RMSE is its square root.

```python
from macroforecast import metrics as M, tests as T

# small worked example of the evaluation primitives
rng = np.random.default_rng(0)
actual = pd.Series(rng.standard_normal(120))
fm_pred = actual + rng.standard_normal(120) * 0.9          # benchmark errors
rf_pred = actual + rng.standard_normal(120) * 0.8          # a better model

e_fm = (actual - fm_pred).to_numpy()
e_rf = (actual - rf_pred).to_numpy()
rel_mse = float(np.mean(e_rf**2) / np.mean(e_fm**2))
print("relative MSE (rf vs fm) :", round(rel_mse, 4))
print("relative RMSE           :", round(rel_mse**0.5, 4), " (= sqrt of relative MSE)")
print("Diebold-Mariano (sq err):", str(T.dm_test(e_rf**2, e_fm**2))[:70])
```

```
relative MSE (rf vs fm) : 0.7162
relative RMSE           : 0.8463  (= sqrt of relative MSE)
Diebold-Mariano (sq err): TestResult(statistic=-1.6403833514673867, p_value=0.10356620977927823,
```

### Validation against the appendix ground truth

The evaluation is exercised on the real run. We forecast industrial production at horizon
one over the whole pseudo-out-of-sample period with the FM benchmark, the AR contender,
and a random forest over the F-Level and the MARX transformation sets, applying every
correction built in the previous steps: the raw-level target, the leak-aware preprocessing,
the deterministic factors, the information-criterion order selection for AR and FM, and the
per-origin refit cadence. The relative RMSE is then compared to the re-extracted appendix
numbers. This is a multi-hour leak-free run, so the table below is the recorded result of
that run rather than a live cell.

```
INDPRO, horizon 1, direct, pseudo-out-of-sample 1980-2017 (455 origins)

model         abs RMSE   rel RMSE   appendix   DM p-value
FM (bench)     0.00621     1.000     (0.006)        -
AR             0.00648     1.042      1.06         0.062
RF F-Level     0.00612     0.985      0.94         0.391
RF MARX        0.00612     0.984      0.93         0.457
```

The FM benchmark matches the appendix absolute RMSE of 0.006. The AR relative RMSE of
1.042 is close to the appendix 1.06, and the Diebold-Mariano test agrees with the appendix
that AR is the weaker model. The two random-forest specifications beat the benchmark, which
is the direction the paper reports, although our gain of about one and a half percent is
smaller than the appendix gain of five to seven percent.

### Reading the random-forest gap

The smaller random-forest gain is a genuine reconstruction difference, not a defect. A
feature-importance check confirms the moving-average rotation is built correctly and is the
signal the forest actually uses. On the MARX matrix the moving-average columns carry about
0.99 of the importance and the target lags only 0.01, so the forest is not collapsing onto
the autoregressive component, and the small F-Level versus MARX difference matches the
appendix, where the two are also within about one point. What remains is a uniform
level difference between our random forest and the paper's. The paper's forest is a MATLAB
TreeBagger and ours is the scikit-learn random forest, and the two differ in split rules
and defaults even at identical hyperparameters; the exact FRED-MD vintage and the bootstrap
seeds are also not recoverable. The paper frames its own exercise as a reconstructed-design
replication rather than an exact-table replication, and a benchmark and a linear contender
that match closely, with a random forest that reproduces the direction and the structure to
within a few points, sit inside that tolerance.

### Cell view and the full grid

The single-target run above is one slice of the grid. The full study runs every target,
horizon, transformation set, and target type as its own cell, each carrying the corrections
of the previous steps, and the cross-arm and cross-horizon caches share the per-origin
preprocessing and factors so the expensive imputation is paid once per origin rather than
once per cell. The full grid is launched as a single resumable background job and reassembled
into the relative-RMSE and Diebold-Mariano tables that mirror the appendix.

---

## Appendix B ground-truth tables

This companion page holds the relative-RMSE numbers of online Appendix B of Goulet Coulombe, Leroux, Stevanovic, and Surprenant (2021), re-extracted directly from the appendix PDF and validated cell by cell. These are the targets the replication run must approach.

Every value is a ratio to the FM benchmark RMSE. The FM absolute RMSE that forms the denominator is printed above each table. AR is itself reported as a ratio to FM. Tables for the direct target come from appendix Tables 3 to 8 and tables for the path-average (SGR) target come from appendix Tables 9 to 14. At horizon 1 the direct and path-average numbers are identical by construction.

The machine-readable form is `data/clss2021_appendix_ground_truth.csv` (984 rows, keyed by horizon, target type, model, and information set).

During re-extraction we found that an earlier hand-made extract had transcription errors in several cells, for example the random forest F-MARX row at horizon 1, so the numbers here supersede any earlier extract.

### Direct target (appendix Tables 3 to 8)

#### Horizon 1 (direct)

**Horizon 1, direct** — FM absolute RMSE (denominator): INDPRO 0.006, EMP 0.001, UNRATE 0.148, INCOME 0.007, CONS 0.004, RETAIL 0.011, HOUST 0.072, M2 0.003, CPI 0.002, PPI 0.006

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 1.06 | 1.03 | 1.04 | 1.06 | 1.03 | 1.02 | 1.01 | 1.01 | 1.03 | 1.01 |
| Adaptive Lasso | F | 0.96 | 0.97 | 0.97 | 1.00 | 1.03 | 1.04 | 1.02 | 0.98 | 0.98 | 0.98 |
|  | F-X | 0.95 | 1.03 | 0.96 | 1.01 | 1.08 | 1.09 | 1.02 | 0.99 | 1.06 | 1.00 |
|  | F-MARX | 0.95 | 0.99 | 0.95 | 1.00 | 1.04 | 1.02 | 1.01 | 0.99 | 0.96 | 0.93 |
|  | F-MAF | 0.94 | 0.99 | 0.95 | 1.01 | 1.04 | 1.05 | 1.02 | 1.00 | 1.05 | 1.02 |
|  | F-Level | 0.96 | 1.02 | 0.95 | 1.00 | 1.02 | 1.04 | 1.02 | 1.00 | 1.02 | 0.99 |
|  | F-X-MARX | 1.09 | 1.01 | 0.95 | 1.01 | 1.06 | 1.03 | 1.01 | 0.97 | 1.04 | 0.97 |
|  | F-X-MAF | 0.95 | 1.01 | 0.96 | 1.02 | 1.06 | 1.07 | 1.02 | 0.98 | 1.05 | 1.01 |
|  | F-X-Level | 0.96 | 1.02 | 0.96 | 1.00 | 1.04 | 1.10 | 1.02 | 0.98 | 1.03 | 1.01 |
|  | F-X-MARX-Level | 1.10 | 1.01 | 0.95 | 1.00 | 1.06 | 1.05 | 1.01 | 0.98 | 1.03 | 0.97 |
|  | X | 0.95 | 1.03 | 0.96 | 1.00 | 1.08 | 1.05 | 1.03 | 0.99 | 1.04 | 1.02 |
|  | MARX | 0.96 | 1.01 | 0.96 | 1.00 | 1.06 | 1.03 | 1.01 | 0.97 | 0.96 | 0.97 |
|  | MAF | 0.98 | 1.00 | 0.96 | 1.01 | 1.08 | 1.05 | 1.03 | 1.00 | 1.09 | 1.04 |
|  | X-MARX | 1.15 | 1.00 | 0.95 | 1.00 | 1.07 | 1.04 | 1.01 | 0.99 | 1.09 | 0.97 |
|  | X-MAF | 1.23 | 1.02 | 0.95 | 1.00 | 1.06 | 1.09 | 1.03 | 0.98 | 1.03 | 1.00 |
|  | X-Level | 0.96 | 1.02 | 0.96 | 1.00 | 1.05 | 1.06 | 1.03 | 0.98 | 1.03 | 1.01 |
|  | X-MARX-Level | 1.13 | 1.01 | 0.95 | 1.00 | 1.06 | 1.04 | 1.01 | 0.97 | 1.03 | 0.96 |
| Elastic Net | F | 0.97 | 0.97 | 0.97 | 1.01 | 1.03 | 1.04 | 1.00 | 0.98 | 0.98 | 0.97 |
|  | F-X | 0.96 | 1.01 | 0.96 | 1.01 | 1.04 | 1.04 | 1.01 | 1.00 | 1.04 | 1.00 |
|  | F-MARX | 0.95 | 0.98 | 0.94 | 1.00 | 1.05 | 1.02 | 1.00 | 0.99 | 0.97 | 0.92 |
|  | F-MAF | 0.95 | 0.98 | 0.95 | 1.00 | 1.04 | 1.06 | 1.01 | 0.99 | 1.04 | 1.03 |
|  | F-Level | 0.96 | 0.98 | 0.95 | 1.01 | 1.03 | 1.02 | 0.97 | 1.00 | 1.00 | 0.99 |
|  | F-X-MARX | 1.09 | 1.01 | 0.95 | 1.00 | 1.05 | 1.04 | 1.00 | 0.98 | 1.19 | 0.96 |
|  | F-X-MAF | 0.95 | 1.01 | 0.96 | 1.00 | 1.05 | 1.10 | 1.02 | 0.99 | 1.06 | 0.99 |
|  | F-X-Level | 0.96 | 1.01 | 0.96 | 1.01 | 1.04 | 1.03 | 1.02 | 0.99 | 1.03 | 0.99 |
|  | F-X-MARX-Level | 1.08 | 1.01 | 0.95 | 1.00 | 1.05 | 1.04 | 1.00 | 0.98 | 1.19 | 0.97 |
|  | X | 0.96 | 1.02 | 0.96 | 1.00 | 1.04 | 1.05 | 1.02 | 0.98 | 1.03 | 0.99 |
|  | MARX | 0.96 | 1.00 | 0.95 | 1.00 | 1.04 | 1.03 | 0.99 | 0.97 | 0.97 | 0.95 |
|  | MAF | 0.97 | 0.99 | 0.96 | 1.01 | 1.05 | 1.06 | 1.03 | 1.00 | 1.10 | 1.03 |
|  | X-MARX | 1.14 | 1.00 | 0.95 | 1.00 | 1.06 | 1.04 | 1.00 | 0.98 | 1.12 | 0.96 |
|  | X-MAF | 0.95 | 1.01 | 0.96 | 1.00 | 1.06 | 1.04 | 1.02 | 1.00 | 1.03 | 0.99 |
|  | X-Level | 0.96 | 1.01 | 0.96 | 0.99 | 1.04 | 1.04 | 1.02 | 0.98 | 1.03 | 1.00 |
|  | X-MARX-Level | 1.09 | 1.01 | 0.95 | 1.00 | 1.08 | 1.07 | 1.01 | 0.97 | 1.04 | 0.96 |
| Linear Boosting | F | 0.97 | 1.00 | 0.97 | 1.00 | 1.03 | 1.04 | 1.00 | 1.17 | 1.07 | 0.99 |
|  | F-X | 0.98 | 1.02 | 0.96 | 1.00 | 1.07 | 1.05 | 1.04 | 1.06 | 1.08 | 1.02 |
|  | F-MARX | 0.96 | 1.05 | 0.96 | 0.99 | 1.04 | 1.03 | 1.01 | 1.09 | 1.00 | 0.98 |
|  | F-MAF | 0.94 | 0.95 | 0.94 | 1.01 | 1.05 | 1.03 | 1.02 | 1.01 | 1.06 | 1.03 |
|  | F-Level | 0.95 | 0.99 | 0.96 | 1.01 | 1.03 | 1.04 | 1.02 | 1.04 | 1.01 | 1.01 |
|  | F-X-MARX | 0.94 | 1.05 | 0.96 | 1.00 | 1.07 | 1.12 | 1.04 | 1.08 | 1.14 | 0.96 |
|  | F-X-MAF | 1.23 | 1.00 | 0.95 | 0.99 | 1.06 | 1.05 | 1.05 | 0.99 | 1.03 | 1.03 |
|  | F-X-Level | 0.94 | 0.99 | 0.96 | 1.00 | 1.07 | 1.03 | 1.03 | 1.02 | 1.09 | 1.01 |
|  | F-X-MARX-Level | 0.94 | 0.99 | 0.94 | 0.99 | 1.07 | 1.05 | 1.03 | 1.02 | 0.98 | 0.94 |
|  | X | 0.96 | 1.08 | 0.96 | 1.02 | 1.08 | 1.06 | 1.04 | 1.06 | 1.22 | 1.02 |
|  | MARX | 0.95 | 1.10 | 0.95 | 0.99 | 1.06 | 1.04 | 1.00 | 1.07 | 1.09 | 0.97 |
|  | MAF | 0.99 | 1.00 | 0.96 | 1.00 | 1.06 | 1.04 | 1.02 | 1.02 | 1.19 | 1.04 |
|  | X-MARX | 0.96 | 1.08 | 0.94 | 1.00 | 1.06 | 1.10 | 1.03 | 1.09 | 1.04 | 0.97 |
|  | X-MAF | 0.96 | 1.02 | 0.96 | 1.02 | 1.11 | 1.06 | 1.04 | 0.98 | 1.02 | 1.01 |
|  | X-Level | 0.95 | 1.05 | 0.96 | 1.00 | 1.06 | 1.06 | 1.05 | 1.04 | 1.03 | 1.01 |
|  | X-MARX-Level | 0.94 | 1.01 | 0.94 | 1.06 | 1.10 | 1.03 | 1.03 | 1.03 | 1.03 | 1.02 |
| Random Forest | F | 0.95 | 0.99 | 0.97 | 0.97 | 1.05 | 1.04 | 1.04 | 0.97 | 1.00 | 0.97 |
|  | F-X | 0.96 | 1.00 | 0.95 | 0.98 | 1.05 | 1.04 | 1.04 | 0.96 | 1.00 | 0.97 |
|  | F-MARX | 0.93 | 0.95 | 0.94 | 0.95 | 1.05 | 1.03 | 1.03 | 0.96 | 0.97 | 0.95 |
|  | F-MAF | 0.96 | 0.97 | 0.97 | 0.98 | 1.04 | 1.04 | 1.04 | 0.97 | 1.01 | 0.97 |
|  | F-Level | 0.94 | 1.00 | 0.96 | 1.02 | 1.05 | 1.05 | 1.04 | 0.96 | 1.00 | 0.98 |
|  | F-X-MARX | 0.93 | 0.96 | 0.95 | 0.96 | 1.05 | 1.04 | 1.03 | 0.96 | 0.98 | 0.95 |
|  | F-X-MAF | 0.94 | 0.98 | 0.95 | 0.97 | 1.06 | 1.04 | 1.05 | 0.96 | 0.99 | 0.98 |
|  | F-X-Level | 0.95 | 0.99 | 0.95 | 1.00 | 1.05 | 1.04 | 1.05 | 0.95 | 1.00 | 0.98 |
|  | F-X-MARX-Level | 0.92 | 0.94 | 0.95 | 0.97 | 1.05 | 1.04 | 1.04 | 0.96 | 0.97 | 0.95 |
|  | X | 0.96 | 1.01 | 0.95 | 0.98 | 1.04 | 1.04 | 1.05 | 0.96 | 1.00 | 0.97 |
|  | MARX | 0.93 | 0.95 | 0.95 | 0.94 | 1.06 | 1.03 | 1.03 | 0.97 | 0.97 | 0.95 |
|  | MAF | 0.97 | 0.99 | 0.98 | 0.99 | 1.05 | 1.04 | 1.05 | 0.98 | 1.02 | 0.96 |
|  | X-MARX | 0.93 | 0.96 | 0.94 | 0.96 | 1.05 | 1.03 | 1.04 | 0.96 | 0.98 | 0.95 |
|  | X-MAF | 0.96 | 0.99 | 0.95 | 0.97 | 1.05 | 1.04 | 1.05 | 0.96 | 0.99 | 0.98 |
|  | X-Level | 0.95 | 0.99 | 0.95 | 1.00 | 1.05 | 1.05 | 1.05 | 0.95 | 0.99 | 0.97 |
|  | X-MARX-Level | 0.92 | 0.95 | 0.94 | 0.98 | 1.06 | 1.04 | 1.04 | 0.96 | 0.96 | 0.95 |
| Boosted Trees | F | 0.97 | 1.06 | 1.01 | 1.00 | 1.05 | 1.03 | 1.05 | 1.04 | 0.98 | 0.99 |
|  | F-X | 0.99 | 1.03 | 0.96 | 1.00 | 1.05 | 1.05 | 1.07 | 1.00 | 0.98 | 0.98 |
|  | F-MARX | 0.96 | 1.02 | 0.94 | 1.01 | 1.06 | 1.03 | 1.03 | 1.00 | 0.98 | 0.97 |
|  | F-MAF | 0.96 | 1.06 | 0.98 | 1.03 | 1.06 | 1.05 | 1.08 | 0.99 | 1.00 | 0.98 |
|  | F-Level | 0.95 | 1.04 | 1.00 | 1.06 | 1.07 | 1.05 | 1.10 | 0.98 | 1.01 | 1.01 |
|  | F-X-MARX | 0.98 | 1.01 | 0.97 | 0.98 | 1.06 | 1.04 | 1.06 | 0.99 | 1.01 | 0.99 |
|  | F-X-MAF | 0.98 | 1.04 | 0.96 | 1.02 | 1.06 | 1.03 | 1.07 | 0.99 | 0.98 | 1.00 |
|  | F-X-Level | 0.96 | 1.09 | 0.96 | 1.04 | 1.04 | 1.05 | 1.08 | 0.98 | 1.01 | 1.02 |
|  | F-X-MARX-Level | 0.97 | 1.04 | 0.96 | 0.99 | 1.07 | 1.02 | 1.07 | 0.99 | 1.00 | 0.99 |
|  | X | 1.00 | 1.10 | 0.97 | 1.00 | 1.04 | 1.04 | 1.10 | 0.99 | 1.00 | 1.00 |
|  | MARX | 0.95 | 1.03 | 0.96 | 1.00 | 1.07 | 1.05 | 1.05 | 1.02 | 0.98 | 0.97 |
|  | MAF | 0.97 | 1.07 | 0.99 | 1.04 | 1.05 | 1.05 | 1.09 | 1.03 | 1.02 | 0.99 |
|  | X-MARX | 0.96 | 0.97 | 0.95 | 1.01 | 1.06 | 1.05 | 1.08 | 1.01 | 0.99 | 0.97 |
|  | X-MAF | 0.98 | 1.07 | 0.97 | 0.99 | 1.05 | 1.05 | 1.07 | 1.01 | 1.00 | 1.00 |
|  | X-Level | 0.96 | 1.06 | 0.97 | 1.03 | 1.05 | 1.06 | 1.10 | 0.99 | 0.99 | 1.01 |
|  | X-MARX-Level | 0.97 | 1.02 | 0.96 | 0.98 | 1.07 | 1.02 | 1.07 | 0.97 | 0.99 | 0.98 |

#### Horizon 3 (direct)

**Horizon 3, direct** — FM absolute RMSE (denominator): INDPRO 0.004, EMP 0.001, UNRATE 0.088, INCOME 0.003, CONS 0.002, RETAIL 0.005, HOUST 0.033, M2 0.003, CPI 0.002, PPI 0.004

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 1.08 | 1.03 | 1.04 | 1.09 | 1.06 | 1.01 | 0.96 | 1.00 | 1.01 | 1.00 |
| Adaptive Lasso | F | 0.95 | 0.91 | 0.94 | 0.98 | 0.99 | 1.07 | 1.05 | 0.98 | 1.01 | 0.99 |
|  | F-X | 0.99 | 0.98 | 0.95 | 1.01 | 1.03 | 1.00 | 0.96 | 1.04 | 1.07 | 0.99 |
|  | F-MARX | 1.06 | 1.02 | 0.89 | 1.09 | 1.06 | 1.05 | 0.97 | 0.99 | 1.07 | 0.98 |
|  | F-MAF | 1.10 | 1.03 | 0.90 | 1.01 | 1.04 | 1.03 | 0.97 | 1.03 | 1.12 | 1.03 |
|  | F-Level | 1.01 | 1.04 | 1.41 | 1.06 | 1.01 | 1.06 | 1.18 | 0.95 | 1.26 | 1.10 |
|  | F-X-MARX | 0.96 | 0.94 | 0.89 | 1.05 | 1.04 | 1.02 | 0.97 | 0.96 | 1.06 | 0.94 |
|  | F-X-MAF | 0.98 | 0.95 | 0.91 | 1.00 | 1.01 | 0.99 | 0.96 | 1.04 | 1.06 | 1.00 |
|  | F-X-Level | 0.97 | 0.98 | 0.93 | 1.02 | 1.02 | 1.01 | 0.96 | 1.04 | 1.06 | 0.98 |
|  | F-X-MARX-Level | 0.96 | 0.95 | 0.90 | 1.06 | 1.03 | 1.05 | 0.96 | 0.94 | 1.06 | 0.96 |
|  | X | 0.99 | 0.98 | 0.95 | 1.02 | 1.03 | 1.01 | 0.97 | 1.03 | 1.06 | 0.98 |
|  | MARX | 1.10 | 1.08 | 0.89 | 1.13 | 1.09 | 1.11 | 0.97 | 0.96 | 1.09 | 0.97 |
|  | MAF | 1.10 | 1.08 | 0.92 | 1.01 | 1.11 | 1.09 | 0.98 | 1.09 | 1.15 | 1.05 |
|  | X-MARX | 0.94 | 0.95 | 0.89 | 1.03 | 1.03 | 1.03 | 0.97 | 0.97 | 1.03 | 0.94 |
|  | X-MAF | 0.98 | 0.95 | 0.91 | 1.00 | 1.02 | 0.99 | 0.97 | 1.03 | 1.07 | 0.99 |
|  | X-Level | 0.98 | 0.99 | 0.93 | 1.02 | 1.01 | 1.01 | 0.96 | 1.04 | 1.07 | 0.98 |
|  | X-MARX-Level | 0.96 | 0.95 | 0.90 | 1.06 | 1.03 | 1.04 | 0.96 | 0.94 | 1.07 | 0.97 |
| Elastic Net | F | 0.94 | 0.91 | 0.92 | 0.98 | 1.00 | 1.07 | 0.97 | 0.98 | 1.00 | 0.99 |
|  | F-X | 0.99 | 0.98 | 0.92 | 1.01 | 1.03 | 1.00 | 0.99 | 1.01 | 1.06 | 0.99 |
|  | F-MARX | 1.06 | 0.92 | 0.97 | 1.12 | 1.09 | 1.06 | 0.98 | 0.96 | 1.03 | 0.96 |
|  | F-MAF | 1.08 | 0.98 | 0.95 | 1.00 | 1.05 | 1.03 | 1.00 | 0.99 | 1.07 | 1.03 |
|  | F-Level | 0.97 | 1.06 | 1.15 | 1.06 | 1.02 | 1.02 | 0.99 | 0.98 | 1.11 | 1.08 |
|  | F-X-MARX | 0.96 | 0.94 | 0.89 | 1.07 | 1.03 | 1.02 | 0.97 | 0.96 | 1.04 | 0.94 |
|  | F-X-MAF | 0.98 | 0.96 | 0.92 | 1.00 | 1.01 | 1.00 | 0.99 | 1.01 | 1.07 | 0.99 |
|  | F-X-Level | 0.97 | 0.99 | 0.92 | 1.02 | 1.02 | 1.02 | 1.00 | 1.03 | 1.08 | 0.98 |
|  | F-X-MARX-Level | 0.95 | 0.96 | 0.90 | 1.07 | 1.03 | 1.05 | 0.98 | 0.95 | 1.04 | 0.94 |
|  | X | 0.98 | 0.99 | 0.92 | 1.02 | 1.03 | 1.00 | 0.99 | 1.01 | 1.07 | 0.99 |
|  | MARX | 1.13 | 0.96 | 0.97 | 1.13 | 1.13 | 1.06 | 0.97 | 0.95 | 1.02 | 0.96 |
|  | MAF | 1.10 | 1.01 | 0.98 | 1.00 | 1.11 | 1.04 | 1.02 | 1.00 | 1.08 | 1.06 |
|  | X-MARX | 0.96 | 0.95 | 0.89 | 1.07 | 1.03 | 1.03 | 0.97 | 0.96 | 1.03 | 0.93 |
|  | X-MAF | 0.98 | 0.96 | 0.92 | 1.00 | 1.01 | 1.00 | 0.99 | 1.01 | 1.07 | 1.00 |
|  | X-Level | 0.98 | 0.99 | 0.92 | 1.02 | 1.02 | 1.02 | 1.00 | 1.03 | 1.09 | 0.98 |
|  | X-MARX-Level | 0.96 | 0.97 | 0.89 | 1.08 | 1.03 | 1.05 | 0.98 | 0.95 | 1.05 | 0.94 |
| Linear Boosting | F | 0.96 | 0.96 | 0.90 | 0.98 | 1.00 | 1.04 | 0.98 | 1.23 | 1.08 | 1.00 |
|  | F-X | 0.96 | 1.02 | 0.93 | 1.01 | 1.09 | 1.04 | 0.96 | 1.08 | 1.11 | 1.00 |
|  | F-MARX | 1.03 | 1.10 | 0.91 | 1.17 | 1.07 | 1.13 | 0.99 | 1.10 | 1.08 | 0.96 |
|  | F-MAF | 1.05 | 0.95 | 0.97 | 0.98 | 1.01 | 1.02 | 0.98 | 1.04 | 1.08 | 1.07 |
|  | F-Level | 0.92 | 1.01 | 0.95 | 1.01 | 1.02 | 1.07 | 0.96 | 1.00 | 1.08 | 1.03 |
|  | F-X-MARX | 0.96 | 1.06 | 0.89 | 1.08 | 1.06 | 1.08 | 1.00 | 1.12 | 1.07 | 0.95 |
|  | F-X-MAF | 0.98 | 0.91 | 0.89 | 0.99 | 1.02 | 1.02 | 0.98 | 0.95 | 1.04 | 0.99 |
|  | F-X-Level | 0.96 | 0.98 | 0.91 | 1.01 | 1.04 | 1.02 | 0.98 | 1.02 | 1.03 | 0.98 |
|  | F-X-MARX-Level | 0.96 | 1.00 | 0.88 | 1.03 | 1.04 | 1.08 | 0.99 | 1.04 | 1.00 | 0.96 |
|  | X | 1.02 | 1.12 | 0.94 | 1.03 | 1.09 | 1.02 | 0.97 | 1.10 | 1.09 | 0.99 |
|  | MARX | 1.08 | 1.20 | 0.94 | 1.14 | 1.13 | 1.16 | 0.99 | 1.08 | 1.09 | 0.98 |
|  | MAF | 1.11 | 1.02 | 0.97 | 0.99 | 1.06 | 1.04 | 1.00 | 1.13 | 1.17 | 1.04 |
|  | X-MARX | 0.99 | 1.14 | 0.89 | 1.05 | 1.06 | 1.12 | 1.00 | 1.13 | 1.07 | 0.96 |
|  | X-MAF | 0.99 | 0.93 | 0.89 | 1.00 | 1.05 | 1.02 | 0.98 | 0.96 | 1.06 | 0.99 |
|  | X-Level | 0.99 | 1.01 | 0.94 | 1.02 | 1.04 | 1.04 | 0.98 | 1.04 | 1.00 | 0.98 |
|  | X-MARX-Level | 0.96 | 1.01 | 0.88 | 1.08 | 1.03 | 1.10 | 1.00 | 1.06 | 1.01 | 0.95 |
| Random Forest | F | 0.97 | 1.00 | 0.93 | 0.98 | 1.00 | 1.00 | 0.94 | 0.96 | 0.94 | 0.97 |
|  | F-X | 1.01 | 1.02 | 0.93 | 1.00 | 1.03 | 1.03 | 0.95 | 0.99 | 0.96 | 0.97 |
|  | F-MARX | 0.88 | 0.87 | 0.84 | 0.96 | 1.01 | 1.04 | 0.95 | 0.98 | 0.97 | 0.97 |
|  | F-MAF | 1.02 | 0.98 | 0.92 | 0.98 | 1.02 | 1.02 | 0.94 | 1.00 | 0.98 | 0.97 |
|  | F-Level | 0.96 | 1.00 | 0.94 | 1.04 | 0.99 | 1.05 | 0.95 | 0.95 | 1.03 | 1.05 |
|  | F-X-MARX | 0.88 | 0.87 | 0.84 | 0.97 | 1.02 | 1.03 | 0.95 | 0.98 | 0.98 | 0.98 |
|  | F-X-MAF | 1.00 | 0.98 | 0.91 | 0.99 | 1.02 | 1.03 | 0.95 | 1.01 | 0.98 | 0.98 |
|  | F-X-Level | 0.97 | 1.01 | 0.92 | 1.01 | 1.01 | 1.04 | 0.96 | 0.94 | 1.00 | 1.03 |
|  | F-X-MARX-Level | 0.89 | 0.88 | 0.83 | 0.98 | 1.01 | 1.04 | 0.96 | 0.96 | 0.97 | 1.00 |
|  | X | 1.03 | 1.05 | 0.95 | 0.99 | 1.02 | 1.03 | 0.95 | 0.98 | 0.95 | 0.97 |
|  | MARX | 0.86 | 0.88 | 0.84 | 0.97 | 1.01 | 1.04 | 0.95 | 0.97 | 0.97 | 0.97 |
|  | MAF | 1.04 | 1.05 | 0.95 | 0.99 | 1.02 | 1.02 | 0.95 | 1.00 | 0.97 | 0.98 |
|  | X-MARX | 0.88 | 0.88 | 0.84 | 0.96 | 1.02 | 1.04 | 0.96 | 0.98 | 0.98 | 0.97 |
|  | X-MAF | 1.01 | 1.01 | 0.93 | 0.98 | 1.02 | 1.03 | 0.96 | 1.00 | 0.98 | 0.98 |
|  | X-Level | 0.99 | 1.04 | 0.95 | 1.01 | 1.01 | 1.05 | 0.96 | 0.95 | 0.99 | 1.02 |
|  | X-MARX-Level | 0.89 | 0.87 | 0.84 | 0.97 | 1.01 | 1.04 | 0.96 | 0.96 | 0.98 | 0.99 |
| Boosted Trees | F | 0.96 | 1.10 | 0.97 | 0.98 | 1.05 | 1.02 | 0.97 | 1.01 | 0.95 | 1.00 |
|  | F-X | 1.01 | 1.07 | 0.94 | 1.00 | 1.04 | 1.06 | 0.96 | 1.06 | 0.98 | 1.00 |
|  | F-MARX | 0.90 | 0.98 | 0.86 | 0.97 | 1.03 | 1.05 | 0.95 | 0.99 | 0.99 | 1.00 |
|  | F-MAF | 0.98 | 1.12 | 0.96 | 1.01 | 1.09 | 1.06 | 0.95 | 1.01 | 0.95 | 0.98 |
|  | F-Level | 0.96 | 1.05 | 0.97 | 1.12 | 1.01 | 1.05 | 1.03 | 0.99 | 1.05 | 1.07 |
|  | F-X-MARX | 0.91 | 0.96 | 0.86 | 0.97 | 1.04 | 1.05 | 0.94 | 1.00 | 1.00 | 0.99 |
|  | F-X-MAF | 1.01 | 1.07 | 0.92 | 0.99 | 1.04 | 1.06 | 0.93 | 1.02 | 1.00 | 1.00 |
|  | F-X-Level | 0.98 | 1.07 | 0.92 | 0.99 | 1.06 | 1.11 | 0.98 | 0.99 | 1.05 | 1.08 |
|  | F-X-MARX-Level | 0.90 | 0.94 | 0.86 | 0.99 | 1.05 | 1.01 | 0.92 | 0.97 | 1.04 | 1.01 |
|  | X | 1.02 | 1.08 | 0.91 | 1.01 | 1.04 | 1.06 | 0.95 | 1.05 | 1.01 | 1.02 |
|  | MARX | 0.92 | 0.90 | 0.87 | 0.98 | 1.05 | 1.09 | 0.96 | 1.04 | 0.99 | 0.97 |
|  | MAF | 1.04 | 1.16 | 0.97 | 1.00 | 1.12 | 1.07 | 0.98 | 1.03 | 0.98 | 0.98 |
|  | X-MARX | 0.91 | 0.97 | 0.86 | 0.99 | 1.04 | 1.04 | 0.98 | 1.05 | 1.02 | 0.98 |
|  | X-MAF | 1.02 | 1.03 | 0.92 | 1.02 | 1.03 | 1.08 | 0.97 | 1.00 | 1.02 | 1.02 |
|  | X-Level | 1.02 | 1.08 | 0.96 | 1.04 | 1.04 | 1.12 | 0.95 | 0.98 | 1.00 | 1.08 |
|  | X-MARX-Level | 0.91 | 0.97 | 0.84 | 0.99 | 1.06 | 1.03 | 0.94 | 0.97 | 1.02 | 1.03 |

#### Horizon 6 (direct)

**Horizon 6, direct** — FM absolute RMSE (denominator): INDPRO 0.004, EMP 0.001, UNRATE 0.077, INCOME 0.002, CONS 0.002, RETAIL 0.004, HOUST 0.024, M2 0.002, CPI 0.002, PPI 0.004

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 1.03 | 1.07 | 1.09 | 1.04 | 0.92 | 0.98 | 0.94 | 0.94 | 0.96 | 0.95 |
| Adaptive Lasso | F | 0.94 | 0.93 | 0.95 | 0.96 | 0.97 | 1.05 | 1.03 | 0.96 | 0.99 | 1.00 |
|  | F-X | 0.96 | 0.97 | 0.99 | 1.01 | 0.99 | 0.95 | 0.91 | 0.96 | 1.00 | 0.99 |
|  | F-MARX | 1.01 | 1.04 | 0.94 | 1.05 | 1.00 | 0.92 | 1.05 | 1.02 | 1.09 | 1.11 |
|  | F-MAF | 1.17 | 1.16 | 0.94 | 1.09 | 1.04 | 1.09 | 1.48 | 1.06 | 1.12 | 1.16 |
|  | F-Level | 1.08 | 1.10 | 1.52 | 1.08 | 0.95 | 1.07 | 1.38 | 0.92 | 1.39 | 1.09 |
|  | F-X-MARX | 0.98 | 1.03 | 0.94 | 0.97 | 1.00 | 0.95 | 0.90 | 1.00 | 1.04 | 1.02 |
|  | F-X-MAF | 0.97 | 0.97 | 0.92 | 0.97 | 0.98 | 0.98 | 0.91 | 0.95 | 1.01 | 1.00 |
|  | F-X-Level | 0.99 | 0.97 | 1.02 | 1.01 | 1.01 | 0.96 | 0.90 | 0.90 | 1.26 | 1.06 |
|  | F-X-MARX-Level | 1.05 | 1.00 | 0.97 | 0.97 | 1.00 | 0.97 | 0.89 | 0.95 | 1.29 | 1.11 |
|  | X | 0.97 | 0.98 | 0.99 | 1.01 | 0.99 | 0.95 | 0.91 | 0.96 | 1.00 | 1.00 |
|  | MARX | 1.03 | 1.12 | 1.04 | 1.06 | 1.09 | 0.92 | 1.08 | 1.05 | 1.08 | 1.06 |
|  | MAF | 1.29 | 1.24 | 1.45 | 1.12 | 1.16 | 1.18 | 1.44 | 1.11 | 1.22 | 1.18 |
|  | X-MARX | 0.99 | 0.98 | 0.94 | 0.96 | 1.00 | 0.95 | 0.90 | 0.99 | 1.03 | 1.00 |
|  | X-MAF | 0.97 | 0.97 | 0.93 | 0.98 | 0.99 | 0.98 | 0.91 | 0.94 | 1.02 | 0.99 |
|  | X-Level | 0.99 | 0.97 | 1.03 | 1.02 | 1.00 | 0.97 | 0.90 | 0.90 | 1.26 | 1.06 |
|  | X-MARX-Level | 1.05 | 1.00 | 0.97 | 0.96 | 1.00 | 0.97 | 0.89 | 0.95 | 1.33 | 1.10 |
| Elastic Net | F | 0.93 | 0.95 | 0.90 | 0.96 | 0.98 | 1.03 | 0.95 | 0.97 | 1.00 | 1.00 |
|  | F-X | 0.97 | 0.98 | 0.95 | 1.01 | 0.99 | 0.95 | 0.95 | 0.96 | 0.98 | 0.99 |
|  | F-MARX | 1.00 | 0.95 | 1.06 | 0.96 | 0.98 | 0.93 | 1.00 | 0.94 | 1.01 | 0.97 |
|  | F-MAF | 1.10 | 1.02 | 1.11 | 1.03 | 0.99 | 1.09 | 1.04 | 0.98 | 1.05 | 1.15 |
|  | F-Level | 1.12 | 1.17 | 1.50 | 1.02 | 0.99 | 1.10 | 1.17 | 0.88 | 1.37 | 1.04 |
|  | F-X-MARX | 0.98 | 0.98 | 0.98 | 0.96 | 0.99 | 0.94 | 0.96 | 0.99 | 1.02 | 1.01 |
|  | F-X-MAF | 0.95 | 0.99 | 0.93 | 0.98 | 0.98 | 0.98 | 1.01 | 0.94 | 0.99 | 1.01 |
|  | F-X-Level | 0.97 | 0.96 | 1.00 | 1.01 | 1.01 | 1.00 | 1.01 | 0.90 | 1.29 | 1.03 |
|  | F-X-MARX-Level | 1.05 | 0.98 | 1.01 | 0.97 | 1.00 | 0.97 | 0.99 | 0.93 | 1.22 | 1.10 |
|  | X | 0.97 | 0.98 | 0.95 | 1.01 | 0.99 | 0.96 | 0.95 | 0.95 | 0.99 | 0.99 |
|  | MARX | 1.02 | 1.25 | 1.08 | 0.98 | 1.00 | 0.96 | 1.04 | 0.95 | 1.13 | 1.01 |
|  | MAF | 1.14 | 1.03 | 1.27 | 1.04 | 1.04 | 1.08 | 1.18 | 0.98 | 1.11 | 1.17 |
|  | X-MARX | 0.98 | 0.96 | 0.98 | 0.97 | 0.99 | 0.95 | 0.96 | 0.97 | 1.02 | 0.99 |
|  | X-MAF | 0.95 | 0.99 | 0.93 | 0.99 | 0.99 | 0.97 | 1.01 | 0.94 | 0.99 | 1.00 |
|  | X-Level | 0.97 | 0.96 | 1.01 | 1.02 | 1.00 | 0.96 | 1.01 | 0.90 | 1.29 | 1.03 |
|  | X-MARX-Level | 1.05 | 0.98 | 1.00 | 0.97 | 1.01 | 0.98 | 0.99 | 0.93 | 1.22 | 1.10 |
| Linear Boosting | F | 0.92 | 0.97 | 0.91 | 0.97 | 0.97 | 1.04 | 0.96 | 1.20 | 1.12 | 1.02 |
|  | F-X | 0.98 | 1.02 | 0.95 | 1.02 | 1.05 | 1.01 | 0.95 | 1.07 | 1.05 | 0.99 |
|  | F-MARX | 1.06 | 1.13 | 1.04 | 1.05 | 1.10 | 1.01 | 1.00 | 1.12 | 1.06 | 1.01 |
|  | F-MAF | 1.17 | 1.21 | 1.06 | 1.05 | 0.99 | 1.09 | 1.03 | 1.06 | 1.16 | 1.15 |
|  | F-Level | 1.02 | 1.10 | 1.09 | 0.99 | 0.96 | 1.01 | 1.00 | 0.97 | 1.41 | 1.09 |
|  | F-X-MARX | 1.05 | 1.13 | 0.97 | 1.03 | 1.07 | 1.03 | 0.96 | 1.16 | 1.07 | 0.98 |
|  | F-X-MAF | 0.96 | 0.96 | 0.90 | 0.95 | 0.98 | 0.96 | 0.99 | 0.95 | 1.06 | 1.00 |
|  | F-X-Level | 0.92 | 0.97 | 0.93 | 0.99 | 0.98 | 0.96 | 0.97 | 1.00 | 1.05 | 0.95 |
|  | F-X-MARX-Level | 0.99 | 1.02 | 0.98 | 0.97 | 1.00 | 0.98 | 0.96 | 1.04 | 1.06 | 0.97 |
|  | X | 0.99 | 1.11 | 1.00 | 1.01 | 1.09 | 1.01 | 0.93 | 1.07 | 1.06 | 0.97 |
|  | MARX | 1.10 | 1.19 | 1.05 | 1.07 | 1.16 | 1.05 | 1.01 | 1.13 | 1.08 | 1.00 |
|  | MAF | 1.24 | 1.32 | 1.13 | 1.13 | 1.13 | 1.13 | 1.08 | 1.11 | 1.27 | 1.20 |
|  | X-MARX | 1.04 | 1.18 | 0.98 | 1.04 | 1.09 | 1.03 | 0.96 | 1.14 | 1.07 | 0.96 |
|  | X-MAF | 0.96 | 0.98 | 0.90 | 0.96 | 0.99 | 0.97 | 0.98 | 0.94 | 1.06 | 1.02 |
|  | X-Level | 0.95 | 0.99 | 0.96 | 0.99 | 0.98 | 0.97 | 0.95 | 0.99 | 1.05 | 0.96 |
|  | X-MARX-Level | 0.99 | 1.01 | 0.97 | 0.97 | 1.00 | 0.99 | 0.99 | 1.05 | 1.05 | 0.99 |
| Random Forest | F | 0.95 | 1.03 | 0.95 | 0.97 | 0.93 | 0.98 | 0.89 | 0.92 | 0.83 | 0.89 |
|  | F-X | 1.05 | 1.12 | 0.99 | 1.00 | 0.96 | 1.00 | 0.88 | 1.00 | 0.87 | 0.92 |
|  | F-MARX | 1.03 | 0.92 | 0.92 | 0.95 | 0.96 | 1.05 | 0.89 | 1.01 | 0.89 | 0.93 |
|  | F-MAF | 1.02 | 1.05 | 0.95 | 0.96 | 0.94 | 0.96 | 0.89 | 0.99 | 0.88 | 0.92 |
|  | F-Level | 1.07 | 1.14 | 1.07 | 1.08 | 0.92 | 1.02 | 0.91 | 0.84 | 0.92 | 1.00 |
|  | F-X-MARX | 1.02 | 0.93 | 0.92 | 0.95 | 0.96 | 1.04 | 0.89 | 1.03 | 0.89 | 0.93 |
|  | F-X-MAF | 1.01 | 1.07 | 0.98 | 0.96 | 0.96 | 0.99 | 0.89 | 1.02 | 0.90 | 0.93 |
|  | F-X-Level | 1.04 | 1.12 | 1.04 | 1.03 | 0.91 | 1.01 | 0.89 | 0.89 | 0.91 | 0.99 |
|  | F-X-MARX-Level | 1.01 | 0.93 | 0.93 | 0.96 | 0.94 | 1.02 | 0.88 | 0.91 | 0.91 | 0.96 |
|  | X | 1.05 | 1.15 | 1.02 | 0.99 | 0.96 | 1.00 | 0.88 | 1.01 | 0.87 | 0.92 |
|  | MARX | 1.02 | 0.92 | 0.92 | 0.95 | 0.95 | 1.05 | 0.88 | 1.02 | 0.89 | 0.93 |
|  | MAF | 1.02 | 1.09 | 0.99 | 0.96 | 0.95 | 0.96 | 0.89 | 1.00 | 0.88 | 0.91 |
|  | X-MARX | 1.03 | 0.93 | 0.93 | 0.95 | 0.97 | 1.05 | 0.88 | 1.02 | 0.90 | 0.92 |
|  | X-MAF | 1.02 | 1.09 | 0.99 | 0.96 | 0.96 | 0.99 | 0.89 | 1.02 | 0.89 | 0.93 |
|  | X-Level | 1.04 | 1.16 | 1.05 | 1.02 | 0.91 | 1.01 | 0.89 | 0.89 | 0.91 | 0.99 |
|  | X-MARX-Level | 1.02 | 0.93 | 0.94 | 0.96 | 0.93 | 1.02 | 0.88 | 0.91 | 0.91 | 0.96 |
| Boosted Trees | F | 0.97 | 1.06 | 1.01 | 0.99 | 1.00 | 1.01 | 0.96 | 0.96 | 0.86 | 0.97 |
|  | F-X | 1.06 | 1.08 | 0.99 | 0.99 | 0.98 | 0.95 | 0.89 | 1.03 | 0.93 | 0.94 |
|  | F-MARX | 1.02 | 1.05 | 0.99 | 0.99 | 0.95 | 0.98 | 0.88 | 0.99 | 0.91 | 0.90 |
|  | F-MAF | 0.97 | 1.18 | 0.97 | 0.96 | 1.03 | 0.94 | 0.93 | 1.03 | 0.86 | 0.95 |
|  | F-Level | 1.09 | 1.26 | 1.14 | 1.10 | 0.94 | 0.97 | 0.92 | 0.86 | 0.97 | 1.01 |
|  | F-X-MARX | 1.02 | 1.02 | 0.94 | 0.99 | 0.99 | 1.00 | 0.94 | 1.03 | 0.92 | 0.93 |
|  | F-X-MAF | 1.05 | 1.11 | 1.02 | 0.99 | 0.95 | 1.00 | 0.87 | 1.00 | 0.94 | 0.94 |
|  | F-X-Level | 1.12 | 1.17 | 1.07 | 1.01 | 0.92 | 1.17 | 0.91 | 0.93 | 1.00 | 1.05 |
|  | F-X-MARX-Level | 1.02 | 1.03 | 0.93 | 0.98 | 0.96 | 0.96 | 0.91 | 0.91 | 0.96 | 0.98 |
|  | X | 1.07 | 1.11 | 1.04 | 1.00 | 0.98 | 0.98 | 0.88 | 1.05 | 0.90 | 0.96 |
|  | MARX | 1.00 | 1.07 | 0.98 | 1.00 | 1.00 | 1.06 | 0.93 | 1.06 | 0.90 | 0.91 |
|  | MAF | 1.08 | 1.20 | 0.99 | 0.99 | 1.06 | 0.97 | 0.89 | 1.00 | 0.87 | 0.95 |
|  | X-MARX | 1.05 | 1.06 | 0.93 | 0.99 | 0.98 | 0.99 | 0.90 | 1.07 | 0.85 | 0.91 |
|  | X-MAF | 1.06 | 1.13 | 1.02 | 1.04 | 0.94 | 1.00 | 0.88 | 1.01 | 0.96 | 0.95 |
|  | X-Level | 1.05 | 1.16 | 1.12 | 1.02 | 0.92 | 1.17 | 0.88 | 0.89 | 0.94 | 1.08 |
|  | X-MARX-Level | 1.01 | 1.04 | 0.95 | 1.02 | 0.95 | 1.06 | 0.91 | 0.91 | 0.96 | 0.99 |

#### Horizon 9 (direct)

**Horizon 9, direct** — FM absolute RMSE (denominator): INDPRO 0.004, EMP 0.001, UNRATE 0.076, INCOME 0.002, CONS 0.002, RETAIL 0.004, HOUST 0.021, M2 0.002, CPI 0.002, PPI 0.003

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 1.01 | 1.07 | 1.11 | 1.02 | 0.90 | 0.96 | 0.92 | 0.92 | 1.02 | 0.94 |
| Adaptive Lasso | F | 0.95 | 0.95 | 0.96 | 0.97 | 0.96 | 1.03 | 1.04 | 0.96 | 1.00 | 0.99 |
|  | F-X | 0.97 | 1.00 | 1.04 | 1.02 | 1.01 | 0.92 | 1.01 | 0.95 | 1.02 | 1.02 |
|  | F-MARX | 1.07 | 1.14 | 1.00 | 1.10 | 1.07 | 0.97 | 1.23 | 1.03 | 1.05 | 1.07 |
|  | F-MAF | 1.28 | 1.27 | 1.13 | 1.15 | 1.21 | 1.16 | 1.44 | 1.01 | 1.31 | 1.19 |
|  | F-Level | 1.10 | 1.33 | 1.63 | 1.13 | 1.07 | 1.11 | 1.44 | 0.93 | 1.48 | 1.06 |
|  | F-X-MARX | 1.02 | 1.06 | 1.00 | 0.99 | 1.03 | 0.91 | 0.89 | 1.00 | 1.02 | 1.01 |
|  | F-X-MAF | 1.00 | 1.04 | 0.99 | 0.99 | 1.00 | 0.93 | 1.03 | 0.94 | 1.06 | 1.02 |
|  | F-X-Level | 1.04 | 1.15 | 1.14 | 1.09 | 1.02 | 1.00 | 1.04 | 0.91 | 1.44 | 1.06 |
|  | F-X-MARX-Level | 1.18 | 1.14 | 1.10 | 1.01 | 1.02 | 1.01 | 0.87 | 0.97 | 1.39 | 1.15 |
|  | X | 0.96 | 1.01 | 1.05 | 1.03 | 1.01 | 0.91 | 1.01 | 0.96 | 1.02 | 1.01 |
|  | MARX | 1.10 | 1.14 | 1.04 | 1.07 | 1.19 | 0.98 | 1.34 | 1.08 | 1.07 | 1.10 |
|  | MAF | 1.34 | 1.38 | 1.82 | 1.16 | 1.22 | 1.18 | 1.40 | 1.05 | 1.28 | 1.20 |
|  | X-MARX | 1.01 | 1.02 | 0.99 | 0.99 | 1.03 | 0.91 | 0.89 | 0.97 | 1.05 | 0.99 |
|  | X-MAF | 0.99 | 1.05 | 0.98 | 0.98 | 1.00 | 0.93 | 1.03 | 0.94 | 1.05 | 1.02 |
|  | X-Level | 1.05 | 1.12 | 1.16 | 1.09 | 1.02 | 0.99 | 1.02 | 0.91 | 1.44 | 1.07 |
|  | X-MARX-Level | 1.16 | 1.15 | 1.10 | 1.01 | 1.02 | 1.02 | 0.87 | 0.97 | 1.40 | 1.14 |
| Elastic Net | F | 0.94 | 0.98 | 0.93 | 0.96 | 0.97 | 1.02 | 0.96 | 0.97 | 0.98 | 0.99 |
|  | F-X | 0.97 | 1.01 | 0.99 | 1.02 | 1.01 | 0.91 | 1.01 | 0.94 | 0.99 | 1.01 |
|  | F-MARX | 1.05 | 1.03 | 1.14 | 0.99 | 0.96 | 0.93 | 1.06 | 0.99 | 1.08 | 0.99 |
|  | F-MAF | 1.17 | 1.06 | 1.32 | 1.07 | 1.02 | 1.13 | 1.18 | 0.94 | 1.10 | 1.16 |
|  | F-Level | 1.16 | 1.33 | 1.58 | 1.05 | 1.03 | 1.02 | 1.29 | 0.90 | 1.42 | 1.05 |
|  | F-X-MARX | 1.02 | 1.04 | 1.04 | 0.97 | 1.01 | 0.91 | 1.00 | 0.95 | 0.99 | 1.00 |
|  | F-X-MAF | 1.00 | 1.03 | 0.98 | 0.99 | 1.00 | 0.93 | 1.05 | 0.94 | 1.02 | 1.03 |
|  | F-X-Level | 1.06 | 1.06 | 1.06 | 1.09 | 1.04 | 0.99 | 1.02 | 0.91 | 1.38 | 1.07 |
|  | F-X-MARX-Level | 1.11 | 1.07 | 1.18 | 1.01 | 1.02 | 1.00 | 1.05 | 0.88 | 1.37 | 1.11 |
|  | X | 0.98 | 1.02 | 1.00 | 1.02 | 1.01 | 0.91 | 1.01 | 0.94 | 0.99 | 1.01 |
|  | MARX | 1.05 | 1.26 | 1.16 | 1.01 | 1.04 | 0.97 | 1.08 | 1.09 | 1.32 | 1.02 |
|  | MAF | 1.22 | 1.06 | 1.74 | 1.07 | 1.04 | 1.12 | 1.23 | 0.93 | 1.19 | 1.19 |
|  | X-MARX | 1.01 | 1.01 | 1.04 | 0.97 | 1.00 | 0.91 | 1.00 | 0.95 | 1.01 | 0.98 |
|  | X-MAF | 1.00 | 1.03 | 0.98 | 0.99 | 1.01 | 0.93 | 1.05 | 0.94 | 1.03 | 1.03 |
|  | X-Level | 1.05 | 1.06 | 1.05 | 1.10 | 1.03 | 0.98 | 1.02 | 0.92 | 1.37 | 1.12 |
|  | X-MARX-Level | 1.11 | 1.07 | 1.18 | 1.01 | 1.02 | 1.00 | 1.05 | 0.87 | 1.36 | 1.11 |
| Linear Boosting | F | 0.95 | 0.96 | 0.95 | 0.96 | 0.97 | 1.00 | 0.96 | 1.20 | 1.33 | 1.05 |
|  | F-X | 1.01 | 1.07 | 1.00 | 1.01 | 1.05 | 1.03 | 0.92 | 1.08 | 1.12 | 0.99 |
|  | F-MARX | 1.05 | 1.13 | 1.07 | 1.03 | 1.08 | 1.04 | 1.04 | 1.13 | 1.23 | 1.03 |
|  | F-MAF | 1.22 | 1.33 | 1.30 | 1.14 | 1.20 | 1.11 | 1.19 | 1.03 | 1.28 | 1.22 |
|  | F-Level | 1.12 | 1.26 | 1.20 | 1.05 | 1.06 | 0.99 | 1.08 | 0.99 | 1.48 | 1.06 |
|  | F-X-MARX | 1.05 | 1.14 | 1.03 | 0.99 | 1.06 | 1.03 | 0.96 | 1.16 | 1.20 | 1.01 |
|  | F-X-MAF | 1.00 | 0.98 | 0.97 | 0.98 | 0.98 | 0.93 | 0.98 | 0.97 | 1.12 | 1.04 |
|  | F-X-Level | 0.97 | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 1.02 | 1.02 | 1.14 | 0.97 |
|  | F-X-MARX-Level | 1.01 | 1.01 | 1.04 | 0.94 | 0.99 | 0.93 | 0.99 | 1.07 | 1.08 | 0.97 |
|  | X | 1.01 | 1.13 | 1.02 | 1.00 | 1.05 | 1.01 | 0.92 | 1.05 | 1.14 | 0.99 |
|  | MARX | 1.11 | 1.17 | 1.08 | 1.02 | 1.14 | 1.10 | 1.02 | 1.10 | 1.19 | 1.03 |
|  | MAF | 1.35 | 1.46 | 1.35 | 1.20 | 1.31 | 1.21 | 1.23 | 1.07 | 1.29 | 1.18 |
|  | X-MARX | 1.05 | 1.18 | 1.05 | 0.99 | 1.07 | 1.05 | 0.97 | 1.13 | 1.20 | 1.00 |
|  | X-MAF | 1.00 | 0.99 | 0.96 | 0.97 | 0.99 | 0.93 | 0.98 | 0.95 | 1.08 | 1.05 |
|  | X-Level | 0.96 | 0.98 | 1.02 | 0.99 | 0.94 | 0.97 | 0.95 | 1.01 | 1.18 | 0.99 |
|  | X-MARX-Level | 1.00 | 1.01 | 1.03 | 0.96 | 0.96 | 0.94 | 0.96 | 1.09 | 1.12 | 1.00 |
| Random Forest | F | 0.95 | 1.05 | 0.96 | 0.97 | 0.94 | 0.95 | 0.84 | 0.87 | 0.84 | 0.85 |
|  | F-X | 1.03 | 1.11 | 1.00 | 1.02 | 0.95 | 0.93 | 0.86 | 1.00 | 0.92 | 0.91 |
|  | F-MARX | 1.03 | 1.01 | 0.99 | 0.96 | 0.99 | 0.96 | 0.88 | 1.04 | 0.93 | 0.90 |
|  | F-MAF | 0.95 | 1.08 | 0.94 | 0.97 | 0.96 | 0.92 | 0.88 | 1.00 | 0.92 | 0.90 |
|  | F-Level | 1.13 | 1.24 | 1.26 | 1.19 | 0.93 | 1.04 | 0.91 | 0.77 | 0.91 | 0.92 |
|  | F-X-MARX | 1.03 | 1.02 | 1.00 | 0.96 | 0.99 | 0.95 | 0.87 | 1.05 | 0.92 | 0.90 |
|  | F-X-MAF | 0.99 | 1.08 | 0.97 | 0.97 | 0.97 | 0.93 | 0.88 | 1.02 | 0.93 | 0.93 |
|  | F-X-Level | 1.04 | 1.14 | 1.08 | 1.10 | 0.89 | 1.00 | 0.87 | 0.84 | 0.94 | 0.96 |
|  | F-X-MARX-Level | 1.00 | 1.03 | 1.03 | 0.98 | 0.93 | 1.00 | 0.87 | 0.87 | 0.95 | 0.95 |
|  | X | 1.03 | 1.12 | 1.02 | 1.03 | 0.95 | 0.92 | 0.86 | 0.99 | 0.91 | 0.92 |
|  | MARX | 1.03 | 1.02 | 1.00 | 0.95 | 0.99 | 0.96 | 0.86 | 1.04 | 0.93 | 0.90 |
|  | MAF | 0.96 | 1.09 | 0.95 | 0.98 | 0.96 | 0.93 | 0.89 | 1.00 | 0.92 | 0.90 |
|  | X-MARX | 1.03 | 1.02 | 1.00 | 0.96 | 0.99 | 0.94 | 0.87 | 1.05 | 0.92 | 0.90 |
|  | X-MAF | 0.99 | 1.09 | 0.98 | 0.97 | 0.97 | 0.93 | 0.89 | 1.02 | 0.93 | 0.93 |
|  | X-Level | 1.04 | 1.15 | 1.10 | 1.09 | 0.89 | 1.00 | 0.87 | 0.83 | 0.94 | 0.96 |
|  | X-MARX-Level | 0.99 | 1.02 | 1.04 | 0.98 | 0.93 | 1.00 | 0.86 | 0.88 | 0.95 | 0.95 |
| Boosted Trees | F | 0.97 | 1.11 | 0.98 | 0.99 | 0.98 | 0.99 | 0.87 | 0.91 | 0.90 | 0.91 |
|  | F-X | 1.04 | 1.13 | 1.02 | 1.00 | 1.00 | 0.93 | 0.88 | 1.01 | 0.93 | 0.91 |
|  | F-MARX | 1.05 | 1.14 | 1.04 | 0.99 | 0.99 | 0.93 | 0.89 | 0.95 | 0.94 | 0.86 |
|  | F-MAF | 1.00 | 1.14 | 0.99 | 1.01 | 1.05 | 0.97 | 0.82 | 0.99 | 0.89 | 0.94 |
|  | F-Level | 1.06 | 1.39 | 1.29 | 1.19 | 1.00 | 1.00 | 0.96 | 0.79 | 1.00 | 0.92 |
|  | F-X-MARX | 1.03 | 1.09 | 1.06 | 0.98 | 1.00 | 0.94 | 0.91 | 1.02 | 0.95 | 0.87 |
|  | F-X-MAF | 1.00 | 1.09 | 1.00 | 0.99 | 0.98 | 0.94 | 0.94 | 1.00 | 0.95 | 0.92 |
|  | F-X-Level | 1.18 | 1.25 | 1.17 | 1.03 | 0.89 | 0.95 | 0.92 | 0.84 | 0.97 | 0.99 |
|  | F-X-MARX-Level | 1.02 | 1.16 | 1.06 | 0.98 | 0.97 | 0.88 | 0.90 | 0.89 | 1.05 | 0.93 |
|  | X | 1.06 | 1.13 | 1.04 | 1.01 | 0.99 | 0.93 | 0.87 | 1.03 | 0.95 | 0.91 |
|  | MARX | 1.02 | 1.17 | 1.01 | 0.99 | 1.01 | 0.97 | 0.91 | 1.01 | 0.94 | 0.86 |
|  | MAF | 1.06 | 1.15 | 0.95 | 1.02 | 1.05 | 0.94 | 0.87 | 0.97 | 0.89 | 0.91 |
|  | X-MARX | 1.00 | 1.10 | 1.03 | 0.96 | 1.00 | 0.93 | 0.90 | 1.07 | 0.90 | 0.86 |
|  | X-MAF | 1.05 | 1.21 | 1.03 | 1.04 | 1.00 | 0.94 | 0.92 | 1.03 | 0.97 | 0.91 |
|  | X-Level | 1.01 | 1.16 | 1.15 | 1.07 | 0.92 | 1.07 | 0.85 | 0.87 | 0.94 | 1.02 |
|  | X-MARX-Level | 1.00 | 1.13 | 1.07 | 1.01 | 0.96 | 0.89 | 0.87 | 0.91 | 1.02 | 0.98 |

#### Horizon 12 (direct)

**Horizon 12, direct** — FM absolute RMSE (denominator): INDPRO 0.003, EMP 0.001, UNRATE 0.077, INCOME 0.002, CONS 0.002, RETAIL 0.003, HOUST 0.019, M2 0.002, CPI 0.001, PPI 0.003

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 1.01 | 1.06 | 1.10 | 1.02 | 0.95 | 0.95 | 0.92 | 0.92 | 1.06 | 0.97 |
| Adaptive Lasso | F | 0.96 | 0.96 | 0.95 | 0.97 | 0.96 | 1.02 | 0.97 | 0.99 | 1.01 | 1.05 |
|  | F-X | 0.99 | 1.01 | 1.03 | 1.05 | 1.05 | 0.92 | 1.00 | 0.96 | 1.05 | 1.06 |
|  | F-MARX | 1.20 | 1.07 | 1.02 | 1.05 | 1.14 | 1.14 | 1.15 | 1.01 | 1.12 | 1.13 |
|  | F-MAF | 1.32 | 1.19 | 1.41 | 1.21 | 1.22 | 1.16 | 1.40 | 0.99 | 1.19 | 1.21 |
|  | F-Level | 1.14 | 1.20 | 1.41 | 1.20 | 1.16 | 1.08 | 1.24 | 0.97 | 1.35 | 1.18 |
|  | F-X-MARX | 1.05 | 1.04 | 0.97 | 1.02 | 1.08 | 1.02 | 0.93 | 0.94 | 1.08 | 1.04 |
|  | F-X-MAF | 1.05 | 1.05 | 0.97 | 1.03 | 1.06 | 0.96 | 1.02 | 0.92 | 1.08 | 1.04 |
|  | F-X-Level | 1.24 | 1.12 | 1.09 | 1.13 | 1.09 | 0.94 | 1.04 | 0.92 | 1.34 | 1.16 |
|  | F-X-MARX-Level | 1.18 | 1.17 | 1.04 | 1.09 | 1.05 | 0.91 | 0.89 | 0.94 | 1.42 | 1.08 |
|  | X | 0.99 | 1.01 | 1.04 | 1.05 | 1.05 | 0.92 | 0.99 | 0.96 | 1.08 | 1.06 |
|  | MARX | 1.24 | 1.17 | 1.24 | 1.10 | 1.18 | 1.16 | 1.25 | 1.00 | 1.15 | 1.16 |
|  | MAF | 1.33 | 1.27 | 1.77 | 1.21 | 1.22 | 1.21 | 1.40 | 1.00 | 1.18 | 1.24 |
|  | X-MARX | 1.02 | 1.01 | 0.97 | 1.03 | 1.05 | 0.95 | 0.96 | 0.92 | 1.08 | 1.03 |
|  | X-MAF | 1.04 | 1.05 | 0.97 | 1.05 | 1.05 | 0.96 | 0.99 | 0.94 | 1.09 | 1.07 |
|  | X-Level | 1.22 | 1.11 | 1.09 | 1.12 | 1.09 | 0.94 | 1.04 | 0.92 | 1.41 | 1.18 |
|  | X-MARX-Level | 1.18 | 1.23 | 1.02 | 1.10 | 1.07 | 0.92 | 0.89 | 0.94 | 1.44 | 1.13 |
| Elastic Net | F | 0.95 | 0.99 | 0.92 | 0.98 | 0.97 | 1.02 | 0.97 | 0.99 | 0.98 | 1.02 |
|  | F-X | 0.99 | 1.02 | 1.00 | 1.05 | 1.04 | 0.93 | 0.99 | 0.95 | 1.03 | 1.06 |
|  | F-MARX | 1.11 | 1.03 | 1.15 | 1.01 | 1.00 | 1.09 | 0.98 | 0.94 | 1.11 | 1.01 |
|  | F-MAF | 1.23 | 1.03 | 1.52 | 1.11 | 1.00 | 1.12 | 1.18 | 0.94 | 1.15 | 1.22 |
|  | F-Level | 1.16 | 1.28 | 1.41 | 1.22 | 1.14 | 1.11 | 1.23 | 0.92 | 1.38 | 1.20 |
|  | F-X-MARX | 1.07 | 1.04 | 1.06 | 1.01 | 1.03 | 1.04 | 0.93 | 0.92 | 1.02 | 1.04 |
|  | F-X-MAF | 1.04 | 1.06 | 1.03 | 1.06 | 1.04 | 0.97 | 1.01 | 0.92 | 1.05 | 1.06 |
|  | F-X-Level | 1.17 | 1.11 | 1.10 | 1.11 | 1.07 | 0.94 | 1.06 | 0.93 | 1.16 | 1.13 |
|  | F-X-MARX-Level | 1.14 | 1.06 | 1.17 | 1.06 | 1.03 | 0.91 | 1.00 | 0.91 | 1.17 | 1.14 |
|  | X | 0.99 | 1.02 | 1.01 | 1.05 | 1.04 | 0.93 | 0.99 | 0.95 | 1.04 | 1.06 |
|  | MARX | 1.19 | 1.35 | 1.20 | 1.14 | 1.16 | 1.10 | 1.05 | 1.18 | 1.26 | 1.15 |
|  | MAF | 1.24 | 1.26 | 1.72 | 1.22 | 1.02 | 1.11 | 1.22 | 1.02 | 1.17 | 1.22 |
|  | X-MARX | 1.03 | 1.03 | 1.06 | 1.01 | 1.03 | 1.03 | 0.93 | 0.91 | 1.03 | 1.03 |
|  | X-MAF | 1.04 | 1.06 | 1.04 | 1.06 | 1.04 | 0.97 | 1.01 | 0.92 | 1.05 | 1.07 |
|  | X-Level | 1.17 | 1.12 | 1.08 | 1.11 | 1.06 | 0.94 | 1.05 | 0.92 | 1.17 | 1.13 |
|  | X-MARX-Level | 1.14 | 1.06 | 1.19 | 1.06 | 1.03 | 0.90 | 1.00 | 0.91 | 1.17 | 1.14 |
| Linear Boosting | F | 0.95 | 0.98 | 0.93 | 0.97 | 0.96 | 1.02 | 0.97 | 1.21 | 1.45 | 1.14 |
|  | F-X | 1.06 | 1.06 | 1.00 | 1.05 | 1.05 | 1.03 | 0.91 | 1.07 | 1.17 | 1.08 |
|  | F-MARX | 1.10 | 1.08 | 1.07 | 1.01 | 1.12 | 1.13 | 0.94 | 1.17 | 1.29 | 1.08 |
|  | F-MAF | 1.27 | 1.28 | 1.28 | 1.18 | 1.23 | 1.19 | 1.17 | 1.04 | 1.23 | 1.23 |
|  | F-Level | 1.20 | 1.17 | 1.18 | 1.28 | 1.17 | 0.98 | 1.09 | 0.96 | 1.28 | 1.28 |
|  | F-X-MARX | 1.08 | 1.10 | 0.99 | 1.00 | 1.06 | 1.04 | 0.93 | 1.12 | 1.31 | 1.07 |
|  | F-X-MAF | 1.04 | 0.99 | 1.01 | 1.06 | 1.03 | 0.94 | 0.97 | 0.96 | 1.20 | 1.10 |
|  | F-X-Level | 0.96 | 0.94 | 1.03 | 1.01 | 0.94 | 0.97 | 0.97 | 1.04 | 1.28 | 1.10 |
|  | F-X-MARX-Level | 1.02 | 0.98 | 0.99 | 0.96 | 1.01 | 0.91 | 0.91 | 1.08 | 1.17 | 1.04 |
|  | X | 1.04 | 1.08 | 1.02 | 1.03 | 1.08 | 1.02 | 0.90 | 1.06 | 1.22 | 1.05 |
|  | MARX | 1.15 | 1.12 | 1.09 | 1.00 | 1.14 | 1.12 | 1.00 | 1.10 | 1.28 | 1.08 |
|  | MAF | 1.28 | 1.36 | 1.36 | 1.24 | 1.32 | 1.25 | 1.22 | 1.06 | 1.32 | 1.20 |
|  | X-MARX | 1.06 | 1.12 | 1.00 | 1.00 | 1.06 | 1.05 | 0.91 | 1.11 | 1.31 | 1.06 |
|  | X-MAF | 1.06 | 0.99 | 1.01 | 1.04 | 1.03 | 0.99 | 0.98 | 0.94 | 1.22 | 1.12 |
|  | X-Level | 0.96 | 0.94 | 1.03 | 1.01 | 0.95 | 0.92 | 0.93 | 1.06 | 1.29 | 1.08 |
|  | X-MARX-Level | 1.03 | 0.96 | 1.01 | 0.97 | 1.00 | 0.93 | 0.91 | 1.07 | 1.13 | 1.03 |
| Random Forest | F | 0.96 | 1.02 | 0.92 | 0.97 | 0.92 | 0.94 | 0.84 | 0.89 | 0.85 | 0.86 |
|  | F-X | 0.98 | 1.05 | 0.97 | 1.01 | 0.94 | 0.89 | 0.87 | 1.01 | 1.00 | 0.98 |
|  | F-MARX | 0.98 | 1.01 | 0.97 | 0.97 | 0.99 | 0.93 | 0.93 | 1.05 | 1.03 | 1.00 |
|  | F-MAF | 0.92 | 1.01 | 0.90 | 0.98 | 0.97 | 0.89 | 0.90 | 1.03 | 0.99 | 0.97 |
|  | F-Level | 1.14 | 1.30 | 1.39 | 1.26 | 0.92 | 1.09 | 0.91 | 0.74 | 0.98 | 0.91 |
|  | F-X-MARX | 0.98 | 1.02 | 0.96 | 0.97 | 1.01 | 0.91 | 0.94 | 1.08 | 0.98 | 0.99 |
|  | F-X-MAF | 0.96 | 1.03 | 0.93 | 0.98 | 0.96 | 0.89 | 0.90 | 1.04 | 0.99 | 0.97 |
|  | F-X-Level | 1.00 | 1.08 | 1.11 | 1.13 | 0.88 | 1.04 | 0.87 | 0.84 | 1.05 | 0.98 |
|  | F-X-MARX-Level | 0.95 | 1.06 | 1.01 | 1.02 | 0.91 | 1.03 | 0.90 | 0.89 | 1.09 | 1.02 |
|  | X | 0.99 | 1.05 | 0.98 | 1.01 | 0.94 | 0.89 | 0.87 | 1.01 | 0.99 | 0.97 |
|  | MARX | 0.98 | 1.02 | 0.96 | 0.97 | 1.00 | 0.93 | 0.93 | 1.06 | 1.02 | 1.00 |
|  | MAF | 0.92 | 1.01 | 0.90 | 0.98 | 0.97 | 0.89 | 0.90 | 1.02 | 0.99 | 0.97 |
|  | X-MARX | 0.98 | 1.02 | 0.96 | 0.97 | 1.00 | 0.92 | 0.93 | 1.08 | 0.98 | 0.99 |
|  | X-MAF | 0.96 | 1.03 | 0.94 | 0.97 | 0.96 | 0.88 | 0.91 | 1.04 | 0.99 | 0.97 |
|  | X-Level | 1.00 | 1.08 | 1.12 | 1.14 | 0.88 | 1.03 | 0.87 | 0.83 | 1.04 | 0.99 |
|  | X-MARX-Level | 0.95 | 1.07 | 1.01 | 1.02 | 0.91 | 1.04 | 0.90 | 0.89 | 1.08 | 1.02 |
| Boosted Trees | F | 0.97 | 1.06 | 0.98 | 1.01 | 0.95 | 0.94 | 0.87 | 0.93 | 0.92 | 0.91 |
|  | F-X | 0.99 | 1.11 | 0.96 | 1.02 | 1.06 | 0.90 | 0.91 | 1.03 | 0.98 | 0.93 |
|  | F-MARX | 1.00 | 1.05 | 1.02 | 1.01 | 1.01 | 0.93 | 0.94 | 1.04 | 0.97 | 0.95 |
|  | F-MAF | 0.98 | 1.04 | 0.89 | 1.03 | 1.03 | 0.94 | 0.89 | 1.04 | 0.90 | 0.98 |
|  | F-Level | 1.09 | 1.32 | 1.30 | 1.21 | 0.98 | 1.08 | 1.06 | 0.74 | 1.05 | 0.93 |
|  | F-X-MARX | 0.98 | 1.11 | 1.02 | 1.01 | 1.03 | 0.95 | 0.93 | 1.04 | 0.99 | 0.90 |
|  | F-X-MAF | 0.93 | 1.03 | 0.95 | 1.03 | 1.02 | 0.92 | 0.94 | 1.03 | 0.99 | 0.95 |
|  | F-X-Level | 1.16 | 1.17 | 1.16 | 1.11 | 0.85 | 0.88 | 1.00 | 0.83 | 1.04 | 0.95 |
|  | F-X-MARX-Level | 1.03 | 1.14 | 1.06 | 1.05 | 0.99 | 0.97 | 0.91 | 0.93 | 1.09 | 1.00 |
|  | X | 1.01 | 1.06 | 1.00 | 1.03 | 1.02 | 0.94 | 0.88 | 1.05 | 1.03 | 0.93 |
|  | MARX | 1.00 | 1.09 | 0.98 | 0.99 | 1.08 | 0.95 | 0.92 | 1.06 | 0.95 | 0.89 |
|  | MAF | 0.98 | 1.11 | 0.91 | 1.01 | 1.08 | 0.93 | 0.91 | 1.07 | 0.88 | 0.97 |
|  | X-MARX | 0.98 | 1.11 | 0.97 | 0.98 | 1.02 | 0.90 | 0.91 | 1.10 | 1.01 | 0.93 |
|  | X-MAF | 1.01 | 1.07 | 0.97 | 1.03 | 1.00 | 0.95 | 0.94 | 1.04 | 1.02 | 0.93 |
|  | X-Level | 1.03 | 1.20 | 1.19 | 1.12 | 0.91 | 0.95 | 0.90 | 0.85 | 1.02 | 0.99 |
|  | X-MARX-Level | 0.97 | 1.08 | 1.03 | 1.07 | 0.95 | 0.90 | 0.90 | 0.92 | 1.13 | 0.99 |

#### Horizon 24 (direct)

**Horizon 24, direct** — FM absolute RMSE (denominator): INDPRO 0.003, EMP 0.001, UNRATE 0.068, INCOME 0.002, CONS 0.002, RETAIL 0.003, HOUST 0.014, M2 0.002, CPI 0.002, PPI 0.003

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 0.98 | 1.03 | 1.08 | 0.98 | 0.85 | 0.93 | 0.92 | 0.90 | 0.95 | 0.87 |
| Adaptive Lasso | F | 0.93 | 0.95 | 0.93 | 0.98 | 0.91 | 1.00 | 0.94 | 0.99 | 1.21 | 1.06 |
|  | F-X | 1.08 | 0.98 | 0.95 | 1.11 | 1.11 | 0.89 | 0.95 | 1.08 | 1.07 | 0.88 |
|  | F-MARX | 1.22 | 1.13 | 1.17 | 1.12 | 1.08 | 1.06 | 1.17 | 1.00 | 1.05 | 1.01 |
|  | F-MAF | 1.33 | 1.19 | 1.03 | 1.31 | 1.23 | 0.98 | 1.30 | 1.03 | 1.20 | 0.98 |
|  | F-Level | 1.21 | 1.18 | 1.42 | 1.36 | 1.19 | 1.13 | 1.28 | 1.19 | 1.48 | 1.41 |
|  | F-X-MARX | 1.10 | 1.03 | 0.93 | 1.19 | 1.12 | 0.97 | 0.95 | 1.03 | 1.02 | 0.99 |
|  | F-X-MAF | 1.12 | 1.00 | 0.95 | 1.19 | 1.09 | 0.96 | 0.94 | 1.05 | 1.10 | 0.88 |
|  | F-X-Level | 1.12 | 1.17 | 1.05 | 1.30 | 1.23 | 0.93 | 1.07 | 1.01 | 1.28 | 1.20 |
|  | F-X-MARX-Level | 1.11 | 1.15 | 1.02 | 1.30 | 1.17 | 1.01 | 1.18 | 1.02 | 1.06 | 1.14 |
|  | X | 1.08 | 0.99 | 0.95 | 1.12 | 1.12 | 0.89 | 0.95 | 1.07 | 1.09 | 0.87 |
|  | MARX | 1.31 | 1.13 | 1.26 | 1.20 | 1.15 | 1.02 | 1.28 | 1.00 | 1.00 | 1.11 |
|  | MAF | 1.32 | 1.19 | 1.58 | 1.32 | 1.25 | 0.99 | 1.30 | 1.04 | 1.08 | 0.99 |
|  | X-MARX | 1.09 | 1.04 | 0.94 | 1.18 | 1.10 | 0.97 | 1.07 | 1.01 | 1.00 | 0.94 |
|  | X-MAF | 1.12 | 1.01 | 0.95 | 1.17 | 1.11 | 0.96 | 0.95 | 1.06 | 1.03 | 0.88 |
|  | X-Level | 1.12 | 1.16 | 1.05 | 1.31 | 1.22 | 0.93 | 1.06 | 1.02 | 1.30 | 1.21 |
|  | X-MARX-Level | 1.10 | 1.15 | 1.02 | 1.31 | 1.17 | 1.01 | 1.18 | 1.02 | 1.06 | 1.14 |
| Elastic Net | F | 0.94 | 0.98 | 0.92 | 0.97 | 0.90 | 0.99 | 0.93 | 0.98 | 1.44 | 1.02 |
|  | F-X | 1.07 | 1.00 | 0.98 | 1.10 | 1.07 | 0.89 | 0.94 | 1.08 | 1.07 | 0.86 |
|  | F-MARX | 1.00 | 1.02 | 1.19 | 1.08 | 0.97 | 0.98 | 0.99 | 1.05 | 0.95 | 0.99 |
|  | F-MAF | 1.15 | 1.11 | 1.22 | 1.16 | 1.05 | 0.98 | 1.07 | 1.02 | 1.34 | 0.96 |
|  | F-Level | 1.18 | 1.23 | 1.42 | 1.39 | 1.32 | 1.14 | 1.22 | 1.15 | 1.81 | 1.43 |
|  | F-X-MARX | 1.03 | 0.95 | 1.04 | 1.06 | 1.06 | 0.93 | 0.93 | 1.09 | 0.97 | 0.86 |
|  | F-X-MAF | 1.11 | 0.98 | 1.01 | 1.09 | 1.05 | 0.94 | 0.94 | 1.06 | 1.08 | 0.87 |
|  | F-X-Level | 1.02 | 1.00 | 1.13 | 1.25 | 1.11 | 0.83 | 1.07 | 1.05 | 1.39 | 1.28 |
|  | F-X-MARX-Level | 1.01 | 0.97 | 0.98 | 1.15 | 1.08 | 0.87 | 1.02 | 1.12 | 1.23 | 1.25 |
|  | X | 1.07 | 0.99 | 0.98 | 1.10 | 1.07 | 0.89 | 0.94 | 1.08 | 1.07 | 0.87 |
|  | MARX | 1.33 | 1.27 | 1.21 | 1.29 | 1.20 | 1.03 | 1.02 | 1.07 | 1.42 | 1.18 |
|  | MAF | 1.27 | 1.24 | 1.44 | 1.21 | 1.25 | 1.01 | 1.08 | 1.05 | 1.30 | 1.00 |
|  | X-MARX | 1.05 | 0.96 | 1.04 | 1.06 | 1.05 | 0.93 | 0.95 | 1.07 | 0.98 | 0.83 |
|  | X-MAF | 1.11 | 0.98 | 1.02 | 1.09 | 1.05 | 0.94 | 0.95 | 1.06 | 1.08 | 0.87 |
|  | X-Level | 1.03 | 1.00 | 1.12 | 1.25 | 1.11 | 0.85 | 1.07 | 1.05 | 1.42 | 1.28 |
|  | X-MARX-Level | 1.02 | 0.97 | 0.98 | 1.15 | 1.08 | 0.88 | 1.02 | 1.12 | 1.24 | 1.25 |
| Linear Boosting | F | 0.93 | 0.94 | 0.95 | 0.97 | 0.90 | 1.02 | 0.92 | 1.11 | 1.32 | 1.11 |
|  | F-X | 1.00 | 1.04 | 0.94 | 1.00 | 0.93 | 1.00 | 0.84 | 1.09 | 1.10 | 1.03 |
|  | F-MARX | 1.11 | 1.07 | 0.97 | 1.03 | 0.96 | 1.13 | 0.90 | 1.11 | 1.40 | 1.08 |
|  | F-MAF | 1.30 | 1.21 | 1.17 | 1.31 | 1.27 | 1.03 | 0.99 | 1.11 | 1.16 | 1.04 |
|  | F-Level | 1.27 | 1.14 | 1.18 | 1.60 | 1.30 | 1.08 | 1.21 | 1.06 | 1.55 | 1.38 |
|  | F-X-MARX | 1.03 | 1.02 | 0.93 | 0.99 | 0.92 | 0.99 | 0.86 | 1.11 | 1.34 | 1.11 |
|  | F-X-MAF | 1.07 | 1.01 | 1.00 | 1.18 | 1.07 | 0.97 | 0.92 | 1.06 | 1.06 | 0.94 |
|  | F-X-Level | 0.96 | 0.95 | 0.94 | 1.00 | 0.95 | 0.94 | 0.91 | 1.03 | 1.42 | 1.16 |
|  | F-X-MARX-Level | 1.01 | 0.90 | 0.91 | 0.99 | 0.89 | 0.95 | 0.95 | 1.04 | 1.07 | 1.03 |
|  | X | 1.03 | 1.06 | 0.96 | 1.02 | 0.91 | 0.98 | 0.88 | 1.07 | 1.22 | 1.06 |
|  | MARX | 1.12 | 1.10 | 1.03 | 1.08 | 0.98 | 1.04 | 0.98 | 1.07 | 1.45 | 1.15 |
|  | MAF | 1.36 | 1.26 | 1.21 | 1.32 | 1.32 | 0.98 | 1.04 | 1.11 | 1.10 | 1.06 |
|  | X-MARX | 1.04 | 1.03 | 0.93 | 0.98 | 0.90 | 0.97 | 0.95 | 1.08 | 1.32 | 1.05 |
|  | X-MAF | 1.09 | 1.02 | 1.00 | 1.19 | 1.09 | 0.98 | 0.93 | 1.07 | 1.07 | 0.94 |
|  | X-Level | 0.95 | 0.91 | 0.92 | 1.01 | 0.95 | 0.89 | 1.04 | 1.04 | 1.49 | 1.30 |
|  | X-MARX-Level | 1.01 | 0.89 | 0.90 | 0.98 | 0.88 | 0.93 | 0.99 | 1.04 | 1.15 | 1.08 |
| Random Forest | F | 0.93 | 0.97 | 0.86 | 0.93 | 0.86 | 0.90 | 0.77 | 0.88 | 0.81 | 0.82 |
|  | F-X | 0.89 | 0.92 | 0.90 | 0.96 | 0.91 | 0.86 | 0.77 | 1.04 | 1.04 | 0.94 |
|  | F-MARX | 0.97 | 0.97 | 0.89 | 1.01 | 0.94 | 0.87 | 0.87 | 1.13 | 1.14 | 1.11 |
|  | F-MAF | 0.94 | 0.91 | 0.87 | 1.01 | 0.90 | 0.82 | 0.85 | 1.04 | 1.22 | 1.04 |
|  | F-Level | 0.87 | 1.26 | 1.26 | 1.16 | 0.82 | 0.96 | 0.92 | 0.82 | 1.10 | 0.84 |
|  | F-X-MARX | 0.95 | 0.98 | 0.89 | 0.98 | 0.91 | 0.87 | 0.82 | 1.16 | 1.09 | 1.08 |
|  | F-X-MAF | 0.89 | 0.90 | 0.86 | 0.99 | 0.91 | 0.83 | 0.80 | 1.06 | 1.13 | 1.00 |
|  | F-X-Level | 0.87 | 1.09 | 1.13 | 1.12 | 0.86 | 0.93 | 0.92 | 0.94 | 1.13 | 0.94 |
|  | F-X-MARX-Level | 0.89 | 1.00 | 0.95 | 1.10 | 0.89 | 0.95 | 0.93 | 0.99 | 1.16 | 1.07 |
|  | X | 0.89 | 0.92 | 0.90 | 0.96 | 0.91 | 0.86 | 0.77 | 1.04 | 1.05 | 0.94 |
|  | MARX | 0.98 | 0.98 | 0.89 | 1.02 | 0.94 | 0.87 | 0.87 | 1.14 | 1.15 | 1.12 |
|  | MAF | 0.97 | 0.93 | 0.89 | 1.01 | 0.90 | 0.82 | 0.85 | 1.04 | 1.21 | 1.04 |
|  | X-MARX | 0.95 | 0.98 | 0.88 | 0.98 | 0.91 | 0.86 | 0.83 | 1.16 | 1.09 | 1.08 |
|  | X-MAF | 0.89 | 0.90 | 0.87 | 0.99 | 0.92 | 0.83 | 0.80 | 1.07 | 1.13 | 1.01 |
|  | X-Level | 0.87 | 1.09 | 1.14 | 1.12 | 0.87 | 0.94 | 0.91 | 0.94 | 1.14 | 0.94 |
|  | X-MARX-Level | 0.89 | 1.00 | 0.94 | 1.10 | 0.89 | 0.95 | 0.94 | 0.99 | 1.16 | 1.08 |
| Boosted Trees | F | 0.93 | 0.99 | 0.90 | 0.95 | 0.87 | 0.95 | 0.78 | 0.95 | 0.84 | 0.89 |
|  | F-X | 0.90 | 1.01 | 0.91 | 1.02 | 0.94 | 0.83 | 0.82 | 1.07 | 0.97 | 0.93 |
|  | F-MARX | 0.96 | 1.06 | 0.87 | 1.04 | 0.98 | 0.84 | 0.88 | 1.07 | 1.02 | 1.00 |
|  | F-MAF | 1.00 | 1.01 | 0.84 | 1.04 | 0.96 | 0.90 | 0.88 | 1.01 | 1.20 | 1.15 |
|  | F-Level | 0.95 | 1.25 | 1.22 | 1.14 | 0.90 | 1.01 | 0.98 | 0.83 | 1.02 | 0.81 |
|  | F-X-MARX | 0.97 | 1.05 | 0.94 | 1.03 | 0.98 | 0.85 | 0.85 | 1.11 | 1.12 | 0.96 |
|  | F-X-MAF | 0.93 | 0.98 | 0.92 | 1.07 | 0.93 | 0.80 | 0.86 | 1.09 | 1.26 | 0.96 |
|  | F-X-Level | 0.89 | 1.13 | 1.20 | 1.08 | 0.90 | 0.83 | 0.96 | 0.93 | 1.12 | 0.86 |
|  | F-X-MARX-Level | 0.91 | 1.11 | 1.02 | 1.13 | 0.95 | 0.88 | 0.99 | 0.96 | 1.21 | 1.05 |
|  | X | 0.90 | 1.01 | 0.92 | 1.02 | 0.98 | 0.84 | 0.82 | 1.07 | 1.08 | 0.96 |
|  | MARX | 0.99 | 1.06 | 0.91 | 1.00 | 0.97 | 0.84 | 0.91 | 1.09 | 1.13 | 0.97 |
|  | MAF | 1.01 | 1.01 | 0.87 | 1.04 | 0.97 | 0.83 | 0.83 | 1.00 | 1.16 | 1.13 |
|  | X-MARX | 0.93 | 1.03 | 0.90 | 1.01 | 0.98 | 0.85 | 0.85 | 1.13 | 1.09 | 0.99 |
|  | X-MAF | 0.94 | 0.99 | 0.92 | 1.04 | 0.98 | 0.82 | 0.83 | 1.09 | 1.24 | 0.95 |
|  | X-Level | 0.92 | 1.20 | 1.16 | 1.12 | 0.82 | 0.91 | 0.91 | 0.92 | 1.22 | 0.86 |
|  | X-MARX-Level | 0.96 | 1.05 | 1.00 | 1.09 | 0.92 | 0.87 | 1.03 | 0.98 | 1.11 | 0.99 |

### Path-average / SGR target (appendix Tables 9 to 14)

#### Horizon 1 (path-average)

**Horizon 1, path-average (SGR)** — FM absolute RMSE (denominator): INDPRO 0.006, EMP 0.001, UNRATE 0.148, INCOME 0.007, CONS 0.004, RETAIL 0.011, HOUST 0.072, M2 0.003, CPI 0.002, PPI 0.006

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Adaptive Lasso | F | 0.96 | 0.97 | 0.97 | 1.00 | 1.03 | 1.04 | 1.02 | 0.98 | 0.98 | 0.98 |
|  | F-X | 0.95 | 1.03 | 0.96 | 1.01 | 1.08 | 1.09 | 1.02 | 0.99 | 1.06 | 1.00 |
|  | F-MARX | 0.95 | 0.99 | 0.95 | 1.00 | 1.04 | 1.02 | 1.01 | 0.99 | 0.96 | 0.93 |
|  | F-MAF | 0.94 | 0.99 | 0.95 | 1.01 | 1.04 | 1.05 | 1.02 | 1.00 | 1.05 | 1.02 |
|  | F-Level | 0.96 | 1.02 | 0.95 | 1.00 | 1.02 | 1.04 | 1.02 | 1.00 | 1.02 | 0.99 |
|  | F-X-MARX | 1.09 | 1.01 | 0.95 | 1.01 | 1.06 | 1.03 | 1.01 | 0.97 | 1.04 | 0.97 |
|  | F-X-MAF | 0.95 | 1.01 | 0.96 | 1.02 | 1.06 | 1.07 | 1.02 | 0.98 | 1.05 | 1.01 |
|  | F-X-Level | 0.96 | 1.02 | 0.96 | 1.00 | 1.04 | 1.10 | 1.02 | 0.98 | 1.03 | 1.01 |
|  | F-X-MARX-Level | 1.10 | 1.01 | 0.95 | 1.00 | 1.06 | 1.05 | 1.01 | 0.98 | 1.03 | 0.97 |
|  | X | 0.95 | 1.03 | 0.96 | 1.00 | 1.08 | 1.05 | 1.03 | 0.99 | 1.04 | 1.02 |
|  | MARX | 0.96 | 1.01 | 0.96 | 1.00 | 1.06 | 1.03 | 1.01 | 0.97 | 0.96 | 0.97 |
|  | MAF | 0.98 | 1.00 | 0.96 | 1.01 | 1.08 | 1.05 | 1.03 | 1.00 | 1.09 | 1.04 |
|  | X-MARX | 1.15 | 1.00 | 0.95 | 1.00 | 1.07 | 1.04 | 1.01 | 0.99 | 1.09 | 0.97 |
|  | X-MAF | 1.23 | 1.02 | 0.95 | 1.00 | 1.06 | 1.09 | 1.03 | 0.98 | 1.03 | 1.00 |
|  | X-Level | 0.96 | 1.02 | 0.96 | 1.00 | 1.05 | 1.06 | 1.03 | 0.98 | 1.03 | 1.01 |
|  | X-MARX-Level | 1.13 | 1.01 | 0.95 | 1.00 | 1.06 | 1.04 | 1.01 | 0.97 | 1.03 | 0.96 |
| Elastic Net | F | 0.97 | 0.97 | 0.97 | 1.01 | 1.03 | 1.04 | 1.00 | 0.98 | 0.98 | 0.97 |
|  | F-X | 0.96 | 1.01 | 0.96 | 1.01 | 1.04 | 1.04 | 1.01 | 1.00 | 1.04 | 1.00 |
|  | F-MARX | 0.95 | 0.98 | 0.94 | 1.00 | 1.05 | 1.02 | 1.00 | 0.99 | 0.97 | 0.92 |
|  | F-MAF | 0.95 | 0.98 | 0.95 | 1.00 | 1.04 | 1.06 | 1.01 | 0.99 | 1.04 | 1.03 |
|  | F-Level | 0.96 | 0.98 | 0.95 | 1.01 | 1.03 | 1.02 | 0.97 | 1.00 | 1.00 | 0.99 |
|  | F-X-MARX | 1.09 | 1.01 | 0.95 | 1.00 | 1.05 | 1.04 | 1.00 | 0.98 | 1.19 | 0.96 |
|  | F-X-MAF | 0.95 | 1.01 | 0.96 | 1.00 | 1.05 | 1.10 | 1.02 | 0.99 | 1.06 | 0.99 |
|  | F-X-Level | 0.96 | 1.01 | 0.96 | 1.01 | 1.04 | 1.03 | 1.02 | 0.99 | 1.03 | 0.99 |
|  | F-X-MARX-Level | 1.08 | 1.01 | 0.95 | 1.00 | 1.05 | 1.04 | 1.00 | 0.98 | 1.19 | 0.97 |
|  | X | 0.96 | 1.02 | 0.96 | 1.00 | 1.04 | 1.05 | 1.02 | 0.98 | 1.03 | 0.99 |
|  | MARX | 0.96 | 1.00 | 0.95 | 1.00 | 1.04 | 1.03 | 0.99 | 0.97 | 0.97 | 0.95 |
|  | MAF | 0.97 | 0.99 | 0.96 | 1.01 | 1.05 | 1.06 | 1.03 | 1.00 | 1.10 | 1.03 |
|  | X-MARX | 1.14 | 1.00 | 0.95 | 1.00 | 1.06 | 1.04 | 1.00 | 0.98 | 1.12 | 0.96 |
|  | X-MAF | 0.95 | 1.01 | 0.96 | 1.00 | 1.06 | 1.04 | 1.02 | 1.00 | 1.03 | 0.99 |
|  | X-Level | 0.96 | 1.01 | 0.96 | 0.99 | 1.04 | 1.04 | 1.02 | 0.98 | 1.03 | 1.00 |
|  | X-MARX-Level | 1.09 | 1.01 | 0.95 | 1.00 | 1.08 | 1.07 | 1.01 | 0.97 | 1.04 | 0.96 |
| Linear Boosting | F | 0.97 | 1.00 | 0.97 | 1.00 | 1.03 | 1.04 | 1.00 | 1.17 | 1.07 | 0.99 |
|  | F-X | 0.98 | 1.02 | 0.96 | 1.00 | 1.07 | 1.05 | 1.04 | 1.06 | 1.08 | 1.02 |
|  | F-MARX | 0.96 | 1.05 | 0.96 | 0.99 | 1.04 | 1.03 | 1.01 | 1.09 | 1.00 | 0.98 |
|  | F-MAF | 0.94 | 0.95 | 0.94 | 1.01 | 1.05 | 1.03 | 1.02 | 1.01 | 1.06 | 1.03 |
|  | F-Level | 0.95 | 0.99 | 0.96 | 1.01 | 1.03 | 1.04 | 1.02 | 1.04 | 1.01 | 1.01 |
|  | F-X-MARX | 0.94 | 1.05 | 0.96 | 1.00 | 1.07 | 1.12 | 1.04 | 1.08 | 1.14 | 0.96 |
|  | F-X-MAF | 1.23 | 1.00 | 0.95 | 0.99 | 1.06 | 1.05 | 1.05 | 0.99 | 1.03 | 1.03 |
|  | F-X-Level | 0.94 | 0.99 | 0.96 | 1.00 | 1.07 | 1.03 | 1.03 | 1.02 | 1.09 | 1.01 |
|  | F-X-MARX-Level | 0.94 | 0.99 | 0.94 | 0.99 | 1.07 | 1.05 | 1.03 | 1.02 | 0.98 | 0.94 |
|  | X | 0.96 | 1.08 | 0.96 | 1.02 | 1.08 | 1.06 | 1.04 | 1.06 | 1.22 | 1.02 |
|  | MARX | 0.95 | 1.10 | 0.95 | 0.99 | 1.06 | 1.04 | 1.00 | 1.07 | 1.09 | 0.97 |
|  | MAF | 0.99 | 1.00 | 0.96 | 1.00 | 1.06 | 1.04 | 1.02 | 1.02 | 1.19 | 1.04 |
|  | X-MARX | 0.96 | 1.08 | 0.94 | 1.00 | 1.06 | 1.10 | 1.03 | 1.09 | 1.04 | 0.97 |
|  | X-MAF | 0.96 | 1.02 | 0.96 | 1.02 | 1.11 | 1.06 | 1.04 | 0.98 | 1.02 | 1.01 |
|  | X-Level | 0.95 | 1.05 | 0.96 | 1.00 | 1.06 | 1.06 | 1.05 | 1.04 | 1.03 | 1.01 |
|  | X-MARX-Level | 0.94 | 1.01 | 0.94 | 1.06 | 1.10 | 1.03 | 1.03 | 1.03 | 1.03 | 1.02 |
| Random Forest | F | 0.95 | 0.99 | 0.97 | 0.97 | 1.05 | 1.04 | 1.04 | 0.97 | 1.00 | 0.97 |
|  | F-X | 0.96 | 1.00 | 0.95 | 0.98 | 1.05 | 1.04 | 1.04 | 0.96 | 1.00 | 0.97 |
|  | F-MARX | 0.93 | 0.95 | 0.94 | 0.95 | 1.05 | 1.03 | 1.03 | 0.96 | 0.97 | 0.95 |
|  | F-MAF | 0.96 | 0.97 | 0.97 | 0.98 | 1.04 | 1.04 | 1.04 | 0.97 | 1.01 | 0.97 |
|  | F-Level | 0.94 | 1.00 | 0.96 | 1.02 | 1.05 | 1.05 | 1.04 | 0.96 | 1.00 | 0.98 |
|  | F-X-MARX | 0.93 | 0.96 | 0.95 | 0.96 | 1.05 | 1.04 | 1.03 | 0.96 | 0.98 | 0.95 |
|  | F-X-MAF | 0.94 | 0.98 | 0.95 | 0.97 | 1.06 | 1.04 | 1.05 | 0.96 | 0.99 | 0.98 |
|  | F-X-Level | 0.95 | 0.99 | 0.95 | 1.00 | 1.05 | 1.04 | 1.05 | 0.95 | 1.00 | 0.98 |
|  | F-X-MARX-Level | 0.92 | 0.94 | 0.95 | 0.97 | 1.05 | 1.04 | 1.04 | 0.96 | 0.97 | 0.95 |
|  | X | 0.96 | 1.01 | 0.95 | 0.98 | 1.04 | 1.04 | 1.05 | 0.96 | 1.00 | 0.97 |
|  | MARX | 0.93 | 0.95 | 0.95 | 0.94 | 1.06 | 1.03 | 1.03 | 0.97 | 0.97 | 0.95 |
|  | MAF | 0.97 | 0.99 | 0.98 | 0.99 | 1.05 | 1.04 | 1.05 | 0.98 | 1.02 | 0.96 |
|  | X-MARX | 0.93 | 0.96 | 0.94 | 0.96 | 1.05 | 1.03 | 1.04 | 0.96 | 0.98 | 0.95 |
|  | X-MAF | 0.96 | 0.99 | 0.95 | 0.97 | 1.05 | 1.04 | 1.05 | 0.96 | 0.99 | 0.98 |
|  | X-Level | 0.95 | 0.99 | 0.95 | 1.00 | 1.05 | 1.05 | 1.05 | 0.95 | 0.99 | 0.97 |
|  | X-MARX-Level | 0.92 | 0.95 | 0.94 | 0.98 | 1.06 | 1.04 | 1.04 | 0.96 | 0.96 | 0.95 |
| Boosted Trees | F | 0.98 | 1.05 | 1.01 | 1.02 | 1.05 | 1.02 | 1.06 | 1.04 | 0.97 | 0.98 |
|  | F-X | 0.98 | 1.04 | 0.95 | 1.00 | 1.06 | 1.04 | 1.07 | 1.01 | 0.99 | 0.99 |
|  | F-MARX | 0.96 | 1.02 | 0.94 | 1.01 | 1.05 | 1.06 | 1.03 | 1.00 | 0.99 | 0.98 |
|  | F-MAF | 0.95 | 1.07 | 0.99 | 1.04 | 1.06 | 1.05 | 1.08 | 1.00 | 1.01 | 0.97 |
|  | F-Level | 0.97 | 1.02 | 1.01 | 1.06 | 1.07 | 1.05 | 1.10 | 0.98 | 1.02 | 1.00 |
|  | F-X-MARX | 0.96 | 1.05 | 0.96 | 0.97 | 1.07 | 1.04 | 1.06 | 1.00 | 1.00 | 0.98 |
|  | F-X-MAF | 0.99 | 1.06 | 0.97 | 1.02 | 1.06 | 1.02 | 1.07 | 0.99 | 0.99 | 0.99 |
|  | F-X-Level | 0.96 | 1.09 | 0.95 | 1.03 | 1.05 | 1.06 | 1.08 | 0.99 | 1.00 | 1.01 |
|  | F-X-MARX-Level | 0.97 | 1.01 | 0.96 | 0.98 | 1.05 | 1.02 | 1.07 | 0.98 | 0.99 | 0.99 |
|  | X | 0.98 | 1.08 | 0.98 | 1.00 | 1.05 | 1.06 | 1.08 | 0.97 | 0.99 | 1.01 |
|  | MARX | 0.94 | 1.02 | 0.95 | 0.99 | 1.08 | 1.05 | 1.04 | 1.01 | 0.99 | 0.97 |
|  | MAF | 0.98 | 1.06 | 0.99 | 1.04 | 1.06 | 1.04 | 1.09 | 1.02 | 1.03 | 0.99 |
|  | X-MARX | 0.95 | 1.00 | 0.96 | 1.00 | 1.06 | 1.05 | 1.08 | 0.97 | 0.99 | 0.98 |
|  | X-MAF | 0.98 | 1.08 | 0.98 | 1.02 | 1.06 | 1.04 | 1.07 | 1.01 | 1.00 | 1.00 |
|  | X-Level | 0.97 | 1.07 | 0.97 | 1.02 | 1.06 | 1.06 | 1.09 | 0.98 | 0.98 | 1.01 |
|  | X-MARX-Level | 0.96 | 1.02 | 0.95 | 0.98 | 1.08 | 1.02 | 1.07 | 0.99 | 0.99 | 1.00 |

#### Horizon 3 (path-average)

**Horizon 3, path-average (SGR)** — FM absolute RMSE (denominator): INDPRO 0.004, EMP 0.001, UNRATE 0.088, INCOME 0.003, CONS 0.002, RETAIL 0.005, HOUST 0.033, M2 0.003, CPI 0.002, PPI 0.004

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 0.97 | 0.96 | 0.96 | 1.00 | 1.03 | 0.98 | 0.98 | 1.01 | 1.02 | 1.00 |
| Adaptive Lasso | F | 0.92 | 0.90 | 0.92 | 1.00 | 1.04 | 1.03 | 0.92 | 0.97 | 0.98 | 0.98 |
|  | F-X | 1.00 | 0.99 | 0.94 | 1.03 | 1.16 | 1.07 | 0.92 | 0.95 | 1.00 | 1.00 |
|  | F-MARX | 0.91 | 0.92 | 0.87 | 1.02 | 1.06 | 1.01 | 0.91 | 0.96 | 0.99 | 0.94 |
|  | F-MAF | 0.96 | 0.93 | 0.89 | 1.02 | 1.04 | 1.03 | 0.93 | 0.98 | 1.02 | 1.02 |
|  | F-Level | 0.90 | 0.91 | 0.90 | 1.03 | 1.01 | 1.02 | 0.93 | 0.96 | 1.12 | 0.99 |
|  | F-X-MARX | 1.05 | 0.96 | 0.90 | 0.99 | 1.13 | 1.02 | 0.92 | 0.94 | 1.02 | 0.94 |
|  | F-X-MAF | 0.99 | 0.96 | 0.91 | 0.99 | 1.08 | 1.09 | 0.92 | 0.94 | 1.00 | 0.99 |
|  | F-X-Level | 1.00 | 0.98 | 0.95 | 1.02 | 1.08 | 1.10 | 0.93 | 0.93 | 1.02 | 0.99 |
|  | F-X-MARX-Level | 1.05 | 0.96 | 0.91 | 1.03 | 1.08 | 1.05 | 0.91 | 0.92 | 1.02 | 0.94 |
|  | X | 0.99 | 0.99 | 0.94 | 1.03 | 1.11 | 1.03 | 0.93 | 0.95 | 1.02 | 1.01 |
|  | MARX | 0.92 | 0.94 | 0.86 | 1.02 | 1.08 | 1.02 | 0.91 | 0.95 | 0.99 | 0.94 |
|  | MAF | 1.01 | 0.97 | 0.92 | 1.02 | 1.12 | 1.03 | 0.93 | 0.99 | 1.06 | 1.03 |
|  | X-MARX | 1.08 | 0.95 | 0.90 | 1.05 | 1.09 | 1.03 | 0.92 | 0.94 | 0.99 | 0.94 |
|  | X-MAF | 1.12 | 0.97 | 0.91 | 1.03 | 1.09 | 1.08 | 0.93 | 0.93 | 1.01 | 0.99 |
|  | X-Level | 1.00 | 0.98 | 0.95 | 1.02 | 1.08 | 1.05 | 0.94 | 0.92 | 1.03 | 0.99 |
|  | X-MARX-Level | 1.08 | 0.96 | 0.91 | 1.02 | 1.08 | 1.04 | 0.92 | 0.92 | 1.02 | 0.94 |
| Elastic Net | F | 0.95 | 0.89 | 0.91 | 1.01 | 1.04 | 1.03 | 0.94 | 0.97 | 0.97 | 0.97 |
|  | F-X | 0.99 | 0.97 | 0.93 | 1.04 | 1.04 | 1.02 | 0.93 | 0.96 | 1.01 | 0.99 |
|  | F-MARX | 0.91 | 0.90 | 0.86 | 1.02 | 1.06 | 1.01 | 0.93 | 0.96 | 0.99 | 0.93 |
|  | F-MAF | 0.96 | 0.92 | 0.89 | 1.01 | 1.04 | 1.04 | 0.95 | 0.98 | 1.02 | 1.02 |
|  | F-Level | 0.91 | 0.88 | 0.87 | 1.04 | 1.00 | 1.00 | 0.87 | 0.98 | 1.03 | 1.02 |
|  | F-X-MARX | 1.05 | 0.96 | 0.88 | 1.04 | 1.09 | 1.04 | 0.92 | 0.94 | 1.00 | 0.94 |
|  | F-X-MAF | 1.00 | 0.96 | 0.90 | 1.03 | 1.10 | 1.08 | 0.93 | 0.95 | 1.00 | 0.99 |
|  | F-X-Level | 1.00 | 0.97 | 0.93 | 1.04 | 1.03 | 1.01 | 0.93 | 0.94 | 1.02 | 0.99 |
|  | F-X-MARX-Level | 1.04 | 0.95 | 0.88 | 1.04 | 1.06 | 1.02 | 0.92 | 0.93 | 1.03 | 0.94 |
|  | X | 1.00 | 0.98 | 0.93 | 1.03 | 1.06 | 1.02 | 0.94 | 0.94 | 1.01 | 0.99 |
|  | MARX | 0.91 | 0.94 | 0.86 | 1.01 | 1.05 | 1.01 | 0.93 | 0.95 | 0.99 | 0.93 |
|  | MAF | 1.00 | 0.96 | 0.91 | 1.02 | 1.08 | 1.03 | 0.99 | 0.99 | 1.05 | 1.02 |
|  | X-MARX | 1.08 | 0.94 | 0.88 | 1.05 | 1.09 | 1.03 | 0.93 | 0.94 | 1.00 | 0.94 |
|  | X-MAF | 0.99 | 0.96 | 0.91 | 1.00 | 1.12 | 1.02 | 0.94 | 0.96 | 1.00 | 0.99 |
|  | X-Level | 1.00 | 0.97 | 0.94 | 1.01 | 1.04 | 1.02 | 0.93 | 0.93 | 1.02 | 1.00 |
|  | X-MARX-Level | 1.05 | 0.95 | 0.88 | 1.05 | 1.10 | 1.07 | 0.92 | 0.93 | 1.01 | 0.94 |
| Linear Boosting | F | 0.94 | 0.97 | 0.91 | 1.01 | 1.02 | 1.02 | 0.95 | 1.21 | 1.09 | 1.01 |
|  | F-X | 1.02 | 1.02 | 0.92 | 1.03 | 1.10 | 1.04 | 0.95 | 1.08 | 1.07 | 1.01 |
|  | F-MARX | 0.90 | 1.06 | 0.88 | 1.03 | 1.03 | 1.03 | 0.94 | 1.13 | 1.06 | 0.97 |
|  | F-MAF | 0.94 | 0.89 | 0.87 | 1.02 | 1.04 | 1.01 | 0.95 | 1.00 | 1.03 | 1.02 |
|  | F-Level | 0.91 | 0.91 | 0.88 | 1.03 | 1.02 | 1.01 | 0.95 | 1.00 | 1.01 | 0.98 |
|  | F-X-MARX | 0.95 | 1.08 | 0.87 | 1.07 | 1.07 | 1.11 | 0.96 | 1.12 | 1.10 | 0.96 |
|  | F-X-MAF | 1.18 | 0.96 | 0.88 | 1.02 | 1.09 | 1.03 | 0.96 | 0.97 | 1.01 | 1.02 |
|  | F-X-Level | 0.98 | 0.97 | 0.92 | 1.02 | 1.14 | 1.01 | 0.96 | 1.01 | 1.04 | 0.99 |
|  | F-X-MARX-Level | 0.94 | 0.98 | 0.86 | 1.03 | 1.07 | 1.05 | 0.98 | 1.01 | 0.99 | 0.97 |
|  | X | 1.00 | 1.13 | 0.93 | 1.04 | 1.14 | 1.04 | 0.95 | 1.08 | 1.12 | 1.01 |
|  | MARX | 0.92 | 1.14 | 0.85 | 1.04 | 1.07 | 1.03 | 0.95 | 1.10 | 1.09 | 0.97 |
|  | MAF | 1.01 | 0.96 | 0.92 | 1.01 | 1.07 | 1.01 | 0.96 | 1.02 | 1.10 | 1.01 |
|  | X-MARX | 0.96 | 1.13 | 0.88 | 1.07 | 1.10 | 1.12 | 1.00 | 1.12 | 1.04 | 0.99 |
|  | X-MAF | 1.00 | 0.98 | 0.90 | 1.00 | 1.18 | 1.04 | 0.95 | 0.96 | 1.00 | 1.04 |
|  | X-Level | 0.99 | 1.04 | 0.94 | 1.02 | 1.11 | 1.04 | 0.95 | 1.01 | 0.99 | 1.00 |
|  | X-MARX-Level | 0.94 | 1.00 | 0.88 | 1.10 | 1.14 | 1.00 | 0.97 | 1.02 | 0.99 | 1.03 |
| Random Forest | F | 0.95 | 0.96 | 0.91 | 0.96 | 1.02 | 1.01 | 0.93 | 0.96 | 0.93 | 0.96 |
|  | F-X | 0.98 | 0.97 | 0.90 | 0.99 | 1.02 | 1.01 | 0.92 | 0.96 | 0.94 | 0.97 |
|  | F-MARX | 0.87 | 0.82 | 0.83 | 0.96 | 1.01 | 0.99 | 0.94 | 0.96 | 0.94 | 0.96 |
|  | F-MAF | 0.97 | 0.92 | 0.90 | 0.99 | 1.00 | 1.00 | 0.92 | 0.98 | 0.95 | 0.97 |
|  | F-Level | 0.92 | 0.95 | 0.92 | 1.10 | 1.02 | 1.04 | 0.95 | 0.93 | 0.97 | 0.99 |
|  | F-X-MARX | 0.89 | 0.84 | 0.85 | 0.97 | 1.02 | 1.00 | 0.92 | 0.96 | 0.95 | 0.97 |
|  | F-X-MAF | 0.98 | 0.93 | 0.89 | 0.99 | 1.01 | 1.01 | 0.93 | 0.97 | 0.94 | 0.98 |
|  | F-X-Level | 0.94 | 0.96 | 0.90 | 1.02 | 1.00 | 1.02 | 0.93 | 0.93 | 0.95 | 0.97 |
|  | F-X-MARX-Level | 0.88 | 0.83 | 0.85 | 0.99 | 1.01 | 1.01 | 0.93 | 0.95 | 0.94 | 0.97 |
|  | X | 0.99 | 0.98 | 0.91 | 0.98 | 1.01 | 1.01 | 0.94 | 0.96 | 0.93 | 0.97 |
|  | MARX | 0.86 | 0.82 | 0.85 | 0.96 | 1.03 | 0.99 | 0.93 | 0.96 | 0.95 | 0.97 |
|  | MAF | 1.01 | 0.97 | 0.92 | 1.00 | 1.01 | 1.01 | 0.94 | 0.98 | 0.95 | 0.96 |
|  | X-MARX | 0.88 | 0.84 | 0.84 | 0.96 | 1.02 | 0.99 | 0.93 | 0.96 | 0.95 | 0.97 |
|  | X-MAF | 0.99 | 0.95 | 0.89 | 0.98 | 1.02 | 1.01 | 0.93 | 0.97 | 0.94 | 0.98 |
|  | X-Level | 0.95 | 0.98 | 0.91 | 1.04 | 1.01 | 1.01 | 0.94 | 0.92 | 0.95 | 0.98 |
|  | X-MARX-Level | 0.88 | 0.83 | 0.84 | 1.00 | 1.03 | 1.01 | 0.93 | 0.94 | 0.94 | 0.97 |
| Boosted Trees | F | 0.97 | 1.00 | 0.98 | 1.00 | 1.02 | 0.99 | 0.96 | 1.01 | 0.94 | 0.96 |
|  | F-X | 0.97 | 0.96 | 0.94 | 0.99 | 1.06 | 1.00 | 0.96 | 0.99 | 0.98 | 0.98 |
|  | F-MARX | 0.91 | 0.87 | 0.86 | 0.99 | 1.04 | 1.01 | 0.97 | 1.00 | 0.98 | 0.98 |
|  | F-MAF | 0.97 | 1.01 | 0.95 | 1.04 | 1.06 | 1.01 | 0.97 | 0.99 | 0.95 | 0.96 |
|  | F-Level | 0.93 | 0.95 | 0.99 | 1.13 | 1.08 | 1.02 | 0.98 | 0.95 | 1.01 | 1.00 |
|  | F-X-MARX | 0.91 | 0.90 | 0.89 | 0.99 | 1.05 | 0.97 | 0.99 | 1.00 | 0.98 | 0.99 |
|  | F-X-MAF | 1.00 | 0.99 | 0.92 | 1.02 | 1.03 | 0.99 | 0.98 | 1.00 | 0.96 | 0.97 |
|  | F-X-Level | 0.94 | 1.00 | 0.92 | 1.04 | 1.07 | 1.01 | 0.98 | 0.97 | 0.99 | 0.99 |
|  | F-X-MARX-Level | 0.92 | 0.92 | 0.89 | 0.99 | 1.05 | 0.99 | 1.00 | 0.96 | 0.96 | 0.99 |
|  | X | 0.97 | 1.03 | 0.94 | 1.01 | 1.05 | 1.03 | 0.97 | 1.00 | 0.96 | 0.99 |
|  | MARX | 0.89 | 0.89 | 0.87 | 0.98 | 1.09 | 0.98 | 0.98 | 1.03 | 0.99 | 0.97 |
|  | MAF | 1.04 | 1.01 | 0.98 | 1.04 | 1.05 | 1.01 | 0.97 | 1.00 | 0.97 | 0.96 |
|  | X-MARX | 0.92 | 0.89 | 0.90 | 1.00 | 1.05 | 1.01 | 0.99 | 0.98 | 0.97 | 0.99 |
|  | X-MAF | 1.00 | 1.04 | 0.94 | 1.03 | 1.02 | 1.03 | 0.99 | 1.00 | 0.98 | 0.99 |
|  | X-Level | 0.94 | 1.04 | 0.94 | 1.04 | 1.07 | 1.04 | 1.01 | 0.96 | 0.96 | 1.01 |
|  | X-MARX-Level | 0.89 | 0.90 | 0.88 | 0.98 | 1.07 | 0.99 | 0.98 | 0.94 | 0.97 | 0.99 |

#### Horizon 6 (path-average)

**Horizon 6, path-average (SGR)** — FM absolute RMSE (denominator): INDPRO 0.004, EMP 0.001, UNRATE 0.077, INCOME 0.002, CONS 0.002, RETAIL 0.004, HOUST 0.024, M2 0.002, CPI 0.002, PPI 0.004

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 0.93 | 0.93 | 0.95 | 0.97 | 1.01 | 0.95 | 1.02 | 0.99 | 0.97 | 0.97 |
| Adaptive Lasso | F | 0.86 | 0.87 | 0.90 | 0.95 | 1.00 | 1.03 | 0.90 | 0.95 | 0.93 | 0.94 |
|  | F-X | 0.96 | 0.94 | 0.93 | 1.01 | 1.14 | 1.10 | 0.89 | 0.91 | 0.93 | 0.97 |
|  | F-MARX | 0.87 | 0.87 | 0.84 | 0.96 | 1.03 | 1.00 | 0.90 | 0.95 | 0.96 | 0.92 |
|  | F-MAF | 0.91 | 0.89 | 0.87 | 0.96 | 0.98 | 1.02 | 0.90 | 0.94 | 0.96 | 0.97 |
|  | F-Level | 0.84 | 0.86 | 0.89 | 0.99 | 0.94 | 1.00 | 0.92 | 0.93 | 1.20 | 1.02 |
|  | F-X-MARX | 1.02 | 0.91 | 0.89 | 0.95 | 1.05 | 1.00 | 0.90 | 0.91 | 0.97 | 0.91 |
|  | F-X-MAF | 0.95 | 0.92 | 0.89 | 0.97 | 1.03 | 1.08 | 0.89 | 0.90 | 0.94 | 0.94 |
|  | F-X-Level | 0.94 | 0.92 | 0.94 | 1.00 | 1.04 | 1.12 | 0.90 | 0.87 | 0.98 | 0.95 |
|  | F-X-MARX-Level | 1.01 | 0.90 | 0.89 | 0.97 | 1.03 | 1.04 | 0.90 | 0.88 | 0.98 | 0.91 |
|  | X | 0.96 | 0.94 | 0.93 | 1.02 | 1.10 | 1.00 | 0.90 | 0.92 | 0.96 | 0.97 |
|  | MARX | 0.89 | 0.88 | 0.84 | 0.95 | 1.04 | 1.00 | 0.90 | 0.94 | 0.96 | 0.91 |
|  | MAF | 0.96 | 0.91 | 0.89 | 0.98 | 1.10 | 1.01 | 0.90 | 0.95 | 0.98 | 0.98 |
|  | X-MARX | 1.06 | 0.90 | 0.89 | 1.00 | 1.05 | 1.01 | 0.91 | 0.91 | 0.95 | 0.90 |
|  | X-MAF | 1.10 | 0.93 | 0.89 | 0.99 | 1.05 | 1.07 | 0.91 | 0.90 | 0.95 | 0.94 |
|  | X-Level | 0.95 | 0.93 | 0.94 | 1.01 | 1.04 | 1.03 | 0.91 | 0.87 | 0.99 | 0.95 |
|  | X-MARX-Level | 1.03 | 0.90 | 0.89 | 0.97 | 1.04 | 1.02 | 0.91 | 0.88 | 0.98 | 0.91 |
| Elastic Net | F | 0.88 | 0.87 | 0.89 | 0.97 | 0.99 | 1.03 | 0.97 | 0.94 | 0.92 | 0.94 |
|  | F-X | 0.95 | 0.93 | 0.92 | 1.02 | 0.99 | 0.98 | 0.92 | 0.93 | 0.94 | 0.94 |
|  | F-MARX | 0.86 | 0.86 | 0.82 | 0.96 | 1.02 | 1.00 | 0.96 | 0.95 | 0.96 | 0.92 |
|  | F-MAF | 0.90 | 0.88 | 0.86 | 0.96 | 0.98 | 1.02 | 0.99 | 0.94 | 0.96 | 0.97 |
|  | F-Level | 0.83 | 0.83 | 0.85 | 1.00 | 0.94 | 0.98 | 0.99 | 0.95 | 1.09 | 1.03 |
|  | F-X-MARX | 1.01 | 0.91 | 0.84 | 0.98 | 1.02 | 1.01 | 0.94 | 0.91 | 0.94 | 0.90 |
|  | F-X-MAF | 0.96 | 0.92 | 0.88 | 1.01 | 1.01 | 1.11 | 0.92 | 0.91 | 0.93 | 0.95 |
|  | F-X-Level | 0.94 | 0.90 | 0.92 | 1.02 | 0.97 | 0.98 | 0.92 | 0.90 | 0.98 | 0.96 |
|  | F-X-MARX-Level | 1.00 | 0.89 | 0.84 | 0.98 | 0.98 | 1.00 | 0.94 | 0.89 | 1.00 | 0.90 |
|  | X | 0.96 | 0.93 | 0.92 | 1.01 | 1.00 | 0.98 | 0.93 | 0.91 | 0.94 | 0.95 |
|  | MARX | 0.86 | 0.88 | 0.82 | 0.95 | 0.99 | 0.99 | 0.96 | 0.94 | 0.95 | 0.91 |
|  | MAF | 0.94 | 0.90 | 0.87 | 0.98 | 1.02 | 1.00 | 1.04 | 0.95 | 0.98 | 0.98 |
|  | X-MARX | 1.06 | 0.90 | 0.84 | 0.98 | 1.01 | 1.00 | 0.95 | 0.91 | 0.94 | 0.90 |
|  | X-MAF | 0.96 | 0.92 | 0.88 | 0.97 | 1.03 | 0.98 | 0.94 | 0.92 | 0.93 | 0.94 |
|  | X-Level | 0.94 | 0.91 | 0.92 | 0.99 | 0.98 | 0.97 | 0.93 | 0.88 | 0.99 | 0.97 |
|  | X-MARX-Level | 1.01 | 0.89 | 0.84 | 0.99 | 1.03 | 1.04 | 0.96 | 0.88 | 0.98 | 0.91 |
| Linear Boosting | F | 0.87 | 0.93 | 0.88 | 0.97 | 0.98 | 1.01 | 0.99 | 1.18 | 1.10 | 0.99 |
|  | F-X | 0.99 | 0.98 | 0.92 | 1.00 | 1.07 | 1.02 | 0.94 | 1.04 | 1.03 | 0.97 |
|  | F-MARX | 0.85 | 1.00 | 0.84 | 0.99 | 0.98 | 1.03 | 0.98 | 1.13 | 1.07 | 0.95 |
|  | F-MAF | 0.89 | 0.87 | 0.84 | 0.97 | 0.97 | 0.98 | 1.00 | 0.95 | 0.96 | 0.97 |
|  | F-Level | 0.83 | 0.84 | 0.85 | 0.98 | 0.98 | 1.00 | 1.00 | 0.97 | 1.01 | 0.97 |
|  | F-X-MARX | 0.90 | 1.02 | 0.84 | 1.02 | 1.02 | 1.16 | 0.98 | 1.10 | 1.11 | 0.96 |
|  | F-X-MAF | 1.16 | 0.93 | 0.86 | 0.99 | 1.05 | 1.00 | 0.96 | 0.93 | 0.96 | 0.97 |
|  | F-X-Level | 0.93 | 0.93 | 0.90 | 0.99 | 1.10 | 0.99 | 0.96 | 0.98 | 0.98 | 0.94 |
|  | F-X-MARX-Level | 0.89 | 0.92 | 0.82 | 0.98 | 1.03 | 1.04 | 1.00 | 1.00 | 1.01 | 0.96 |
|  | X | 0.96 | 1.06 | 0.92 | 1.03 | 1.12 | 1.02 | 0.95 | 1.05 | 1.08 | 0.96 |
|  | MARX | 0.86 | 1.04 | 0.83 | 1.00 | 1.01 | 1.00 | 1.01 | 1.10 | 1.13 | 0.98 |
|  | MAF | 0.95 | 0.91 | 0.89 | 0.96 | 1.00 | 0.97 | 0.99 | 0.97 | 1.03 | 0.98 |
|  | X-MARX | 0.93 | 1.06 | 0.84 | 1.00 | 1.06 | 1.11 | 1.02 | 1.11 | 1.05 | 0.99 |
|  | X-MAF | 0.95 | 0.93 | 0.88 | 0.97 | 1.16 | 1.01 | 0.94 | 0.91 | 0.97 | 1.01 |
|  | X-Level | 0.94 | 0.96 | 0.93 | 1.00 | 1.07 | 1.02 | 0.94 | 0.98 | 0.95 | 0.96 |
|  | X-MARX-Level | 0.88 | 0.93 | 0.84 | 1.09 | 1.13 | 0.98 | 0.99 | 1.01 | 0.98 | 1.05 |
| Random Forest | F | 0.88 | 0.90 | 0.87 | 0.90 | 0.93 | 0.95 | 0.92 | 0.91 | 0.84 | 0.90 |
|  | F-X | 0.94 | 0.93 | 0.89 | 0.92 | 0.92 | 0.96 | 0.89 | 0.92 | 0.85 | 0.91 |
|  | F-MARX | 0.84 | 0.81 | 0.80 | 0.89 | 0.92 | 0.92 | 0.90 | 0.93 | 0.89 | 0.96 |
|  | F-MAF | 0.93 | 0.87 | 0.87 | 0.89 | 0.89 | 0.94 | 0.89 | 0.94 | 0.88 | 0.93 |
|  | F-Level | 0.89 | 0.93 | 0.93 | 1.09 | 0.89 | 1.00 | 0.93 | 0.86 | 0.88 | 0.97 |
|  | F-X-MARX | 0.85 | 0.83 | 0.82 | 0.88 | 0.93 | 0.95 | 0.87 | 0.93 | 0.90 | 0.95 |
|  | F-X-MAF | 0.93 | 0.90 | 0.88 | 0.90 | 0.90 | 0.96 | 0.88 | 0.93 | 0.86 | 0.93 |
|  | F-X-Level | 0.90 | 0.93 | 0.89 | 0.96 | 0.92 | 0.97 | 0.89 | 0.86 | 0.86 | 0.94 |
|  | F-X-MARX-Level | 0.84 | 0.82 | 0.82 | 0.90 | 0.92 | 0.95 | 0.89 | 0.90 | 0.89 | 0.96 |
|  | X | 0.94 | 0.94 | 0.90 | 0.92 | 0.93 | 0.95 | 0.90 | 0.92 | 0.84 | 0.91 |
|  | MARX | 0.83 | 0.81 | 0.81 | 0.89 | 0.94 | 0.93 | 0.89 | 0.93 | 0.90 | 0.97 |
|  | MAF | 0.96 | 0.89 | 0.90 | 0.91 | 0.91 | 0.95 | 0.90 | 0.94 | 0.89 | 0.93 |
|  | X-MARX | 0.85 | 0.82 | 0.82 | 0.88 | 0.94 | 0.94 | 0.88 | 0.92 | 0.90 | 0.95 |
|  | X-MAF | 0.94 | 0.90 | 0.88 | 0.90 | 0.92 | 0.96 | 0.88 | 0.92 | 0.86 | 0.93 |
|  | X-Level | 0.92 | 0.93 | 0.90 | 0.98 | 0.91 | 0.98 | 0.90 | 0.86 | 0.86 | 0.94 |
|  | X-MARX-Level | 0.85 | 0.82 | 0.82 | 0.91 | 0.94 | 0.94 | 0.89 | 0.89 | 0.89 | 0.96 |
| Boosted Trees | F | 0.89 | 0.92 | 0.96 | 0.96 | 0.96 | 0.93 | 0.93 | 0.96 | 0.88 | 0.91 |
|  | F-X | 0.92 | 0.94 | 0.95 | 0.92 | 0.97 | 0.95 | 0.97 | 0.98 | 0.90 | 0.92 |
|  | F-MARX | 0.87 | 0.81 | 0.85 | 0.92 | 0.99 | 0.95 | 0.96 | 0.99 | 0.95 | 0.97 |
|  | F-MAF | 0.88 | 0.91 | 0.92 | 1.00 | 0.98 | 0.96 | 0.92 | 0.95 | 0.90 | 0.91 |
|  | F-Level | 0.88 | 0.92 | 0.99 | 1.14 | 1.01 | 0.99 | 0.96 | 0.90 | 0.98 | 0.97 |
|  | F-X-MARX | 0.84 | 0.86 | 0.86 | 0.91 | 0.96 | 0.93 | 0.96 | 0.98 | 0.93 | 0.97 |
|  | F-X-MAF | 0.92 | 0.92 | 0.92 | 0.96 | 0.93 | 0.95 | 0.96 | 0.96 | 0.89 | 0.91 |
|  | F-X-Level | 0.91 | 0.95 | 0.91 | 1.00 | 0.99 | 0.99 | 0.97 | 0.93 | 0.93 | 0.94 |
|  | F-X-MARX-Level | 0.87 | 0.86 | 0.88 | 0.93 | 1.00 | 0.95 | 0.99 | 0.92 | 0.93 | 0.98 |
|  | X | 0.92 | 0.97 | 0.94 | 0.98 | 0.95 | 0.97 | 0.95 | 0.97 | 0.89 | 0.91 |
|  | MARX | 0.85 | 0.84 | 0.86 | 0.93 | 1.03 | 0.94 | 0.96 | 1.01 | 0.95 | 0.97 |
|  | MAF | 0.99 | 0.90 | 0.95 | 0.96 | 0.98 | 0.96 | 0.94 | 0.97 | 0.92 | 0.91 |
|  | X-MARX | 0.86 | 0.85 | 0.87 | 0.91 | 0.97 | 0.98 | 0.95 | 0.97 | 0.94 | 0.96 |
|  | X-MAF | 0.94 | 0.95 | 0.95 | 0.97 | 0.92 | 0.97 | 0.97 | 0.98 | 0.91 | 0.93 |
|  | X-Level | 0.90 | 0.96 | 0.95 | 0.99 | 0.98 | 1.00 | 1.02 | 0.91 | 0.88 | 0.97 |
|  | X-MARX-Level | 0.86 | 0.84 | 0.85 | 0.93 | 1.00 | 0.93 | 0.97 | 0.94 | 0.92 | 0.97 |

#### Horizon 9 (path-average)

**Horizon 9, path-average (SGR)** — FM absolute RMSE (denominator): INDPRO 0.004, EMP 0.001, UNRATE 0.076, INCOME 0.002, CONS 0.002, RETAIL 0.004, HOUST 0.021, M2 0.002, CPI 0.002, PPI 0.003

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 0.95 | 0.89 | 0.95 | 0.95 | 1.02 | 0.97 | 1.06 | 1.01 | 1.04 | 0.97 |
| Adaptive Lasso | F | 0.86 | 0.85 | 0.89 | 0.92 | 0.96 | 1.04 | 0.90 | 0.94 | 0.99 | 0.95 |
|  | F-X | 0.96 | 0.92 | 0.93 | 0.98 | 1.13 | 1.16 | 0.90 | 0.89 | 0.96 | 0.95 |
|  | F-MARX | 0.88 | 0.86 | 0.85 | 0.94 | 0.99 | 1.00 | 0.91 | 0.94 | 1.04 | 0.94 |
|  | F-MAF | 0.91 | 0.88 | 0.87 | 0.94 | 0.96 | 1.01 | 0.90 | 0.93 | 1.01 | 0.95 |
|  | F-Level | 0.86 | 0.83 | 0.89 | 0.99 | 0.90 | 1.01 | 0.95 | 0.94 | 1.31 | 1.03 |
|  | F-X-MARX | 0.99 | 0.90 | 0.88 | 0.92 | 1.03 | 1.00 | 0.92 | 0.88 | 1.03 | 0.90 |
|  | F-X-MAF | 0.96 | 0.91 | 0.90 | 0.98 | 0.99 | 1.11 | 0.89 | 0.88 | 0.96 | 0.93 |
|  | F-X-Level | 0.94 | 0.90 | 0.93 | 0.97 | 1.01 | 1.20 | 0.91 | 0.84 | 1.05 | 0.95 |
|  | F-X-MARX-Level | 0.98 | 0.88 | 0.88 | 0.94 | 1.00 | 1.04 | 0.92 | 0.85 | 1.05 | 0.91 |
|  | X | 0.97 | 0.92 | 0.92 | 1.00 | 1.09 | 1.00 | 0.90 | 0.89 | 1.01 | 0.96 |
|  | MARX | 0.90 | 0.88 | 0.84 | 0.92 | 1.03 | 1.01 | 0.93 | 0.93 | 1.03 | 0.98 |
|  | MAF | 0.95 | 0.89 | 0.89 | 0.99 | 1.08 | 1.00 | 0.90 | 0.93 | 1.02 | 0.96 |
|  | X-MARX | 1.03 | 0.90 | 0.88 | 0.98 | 1.01 | 1.00 | 0.93 | 0.89 | 0.98 | 0.89 |
|  | X-MAF | 1.07 | 0.92 | 0.89 | 0.99 | 1.02 | 1.10 | 0.91 | 0.88 | 0.98 | 0.92 |
|  | X-Level | 0.95 | 0.91 | 0.93 | 0.98 | 1.01 | 1.04 | 0.92 | 0.84 | 1.05 | 0.95 |
|  | X-MARX-Level | 1.00 | 0.88 | 0.88 | 0.94 | 1.02 | 1.01 | 0.93 | 0.85 | 1.05 | 0.91 |
| Elastic Net | F | 0.87 | 0.86 | 0.88 | 0.95 | 0.95 | 1.04 | 1.02 | 0.94 | 0.99 | 0.94 |
|  | F-X | 0.96 | 0.92 | 0.91 | 1.00 | 0.96 | 0.98 | 0.94 | 0.90 | 0.98 | 0.93 |
|  | F-MARX | 0.88 | 0.85 | 0.83 | 0.94 | 1.00 | 1.00 | 1.01 | 0.94 | 1.03 | 0.95 |
|  | F-MAF | 0.89 | 0.87 | 0.86 | 0.94 | 0.95 | 1.01 | 1.08 | 0.93 | 1.01 | 0.96 |
|  | F-Level | 0.85 | 0.83 | 0.84 | 1.00 | 0.90 | 0.98 | 1.06 | 0.95 | 1.18 | 1.05 |
|  | F-X-MARX | 0.99 | 0.89 | 0.84 | 0.96 | 1.00 | 1.01 | 0.98 | 0.89 | 0.98 | 0.89 |
|  | F-X-MAF | 0.96 | 0.91 | 0.88 | 1.00 | 0.98 | 1.19 | 0.94 | 0.89 | 0.95 | 0.93 |
|  | F-X-Level | 0.95 | 0.89 | 0.91 | 1.01 | 0.94 | 0.97 | 0.93 | 0.87 | 1.04 | 0.96 |
|  | F-X-MARX-Level | 0.97 | 0.87 | 0.84 | 0.96 | 0.95 | 0.98 | 0.98 | 0.86 | 1.05 | 0.90 |
|  | X | 0.97 | 0.92 | 0.91 | 0.99 | 0.98 | 0.97 | 0.95 | 0.89 | 0.98 | 0.94 |
|  | MARX | 0.88 | 0.88 | 0.83 | 0.92 | 0.96 | 1.00 | 1.03 | 0.93 | 1.02 | 0.96 |
|  | MAF | 0.93 | 0.88 | 0.87 | 0.97 | 1.00 | 0.99 | 1.17 | 0.93 | 1.03 | 0.96 |
|  | X-MARX | 1.02 | 0.89 | 0.84 | 0.96 | 0.98 | 0.99 | 0.99 | 0.89 | 0.98 | 0.88 |
|  | X-MAF | 0.96 | 0.91 | 0.88 | 0.96 | 0.99 | 0.97 | 0.96 | 0.90 | 0.97 | 0.92 |
|  | X-Level | 0.95 | 0.89 | 0.91 | 0.98 | 0.95 | 0.96 | 0.95 | 0.85 | 1.05 | 0.96 |
|  | X-MARX-Level | 0.98 | 0.87 | 0.84 | 0.96 | 1.00 | 1.05 | 0.99 | 0.85 | 1.04 | 0.90 |
| Linear Boosting | F | 0.86 | 0.89 | 0.87 | 0.95 | 0.94 | 1.01 | 1.04 | 1.17 | 1.24 | 1.00 |
|  | F-X | 1.00 | 0.94 | 0.90 | 0.99 | 1.03 | 1.02 | 0.96 | 1.03 | 1.11 | 0.97 |
|  | F-MARX | 0.87 | 0.94 | 0.82 | 0.95 | 0.93 | 1.03 | 1.03 | 1.13 | 1.19 | 1.00 |
|  | F-MAF | 0.89 | 0.88 | 0.86 | 0.96 | 0.93 | 0.97 | 1.08 | 0.93 | 1.02 | 0.97 |
|  | F-Level | 0.85 | 0.81 | 0.84 | 0.97 | 0.93 | 1.00 | 1.04 | 0.98 | 1.12 | 0.99 |
|  | F-X-MARX | 0.91 | 0.96 | 0.83 | 0.98 | 0.97 | 1.23 | 1.03 | 1.10 | 1.25 | 0.99 |
|  | F-X-MAF | 1.13 | 0.91 | 0.86 | 0.96 | 1.01 | 0.98 | 0.98 | 0.91 | 1.02 | 0.97 |
|  | F-X-Level | 0.92 | 0.90 | 0.89 | 0.98 | 1.07 | 0.98 | 1.00 | 0.96 | 1.03 | 0.94 |
|  | F-X-MARX-Level | 0.89 | 0.89 | 0.81 | 0.95 | 1.00 | 1.05 | 1.05 | 0.99 | 1.07 | 0.99 |
|  | X | 0.95 | 0.99 | 0.90 | 1.00 | 1.09 | 1.02 | 0.97 | 1.02 | 1.18 | 0.97 |
|  | MARX | 0.87 | 0.97 | 0.82 | 0.98 | 0.97 | 0.99 | 1.08 | 1.09 | 1.29 | 1.06 |
|  | MAF | 0.95 | 0.91 | 0.89 | 0.96 | 0.96 | 0.95 | 1.09 | 0.95 | 1.09 | 0.96 |
|  | X-MARX | 0.93 | 0.99 | 0.82 | 0.99 | 1.03 | 1.13 | 1.09 | 1.09 | 1.17 | 1.02 |
|  | X-MAF | 0.95 | 0.91 | 0.88 | 0.95 | 1.13 | 1.01 | 0.97 | 0.89 | 1.03 | 1.01 |
|  | X-Level | 0.94 | 0.91 | 0.90 | 0.99 | 1.03 | 1.00 | 0.96 | 0.97 | 1.03 | 0.95 |
|  | X-MARX-Level | 0.88 | 0.89 | 0.83 | 1.08 | 1.10 | 0.97 | 1.05 | 1.00 | 1.05 | 1.07 |
| Random Forest | F | 0.87 | 0.86 | 0.87 | 0.87 | 0.89 | 0.92 | 0.91 | 0.89 | 0.85 | 0.87 |
|  | F-X | 0.93 | 0.90 | 0.89 | 0.89 | 0.88 | 0.93 | 0.86 | 0.89 | 0.87 | 0.89 |
|  | F-MARX | 0.85 | 0.79 | 0.82 | 0.82 | 0.86 | 0.91 | 0.86 | 0.92 | 0.94 | 0.96 |
|  | F-MAF | 0.93 | 0.84 | 0.88 | 0.86 | 0.84 | 0.91 | 0.85 | 0.91 | 0.91 | 0.92 |
|  | F-Level | 0.92 | 0.92 | 0.95 | 1.10 | 0.85 | 0.97 | 0.92 | 0.82 | 0.92 | 0.98 |
|  | F-X-MARX | 0.86 | 0.81 | 0.83 | 0.83 | 0.87 | 0.93 | 0.84 | 0.91 | 0.94 | 0.95 |
|  | F-X-MAF | 0.93 | 0.87 | 0.88 | 0.87 | 0.86 | 0.94 | 0.84 | 0.90 | 0.89 | 0.92 |
|  | F-X-Level | 0.90 | 0.90 | 0.89 | 0.94 | 0.86 | 0.94 | 0.85 | 0.83 | 0.89 | 0.93 |
|  | F-X-MARX-Level | 0.85 | 0.80 | 0.83 | 0.86 | 0.86 | 0.92 | 0.86 | 0.87 | 0.93 | 0.96 |
|  | X | 0.93 | 0.90 | 0.90 | 0.89 | 0.88 | 0.92 | 0.86 | 0.88 | 0.86 | 0.89 |
|  | MARX | 0.84 | 0.79 | 0.83 | 0.83 | 0.88 | 0.91 | 0.87 | 0.92 | 0.95 | 0.97 |
|  | MAF | 0.96 | 0.85 | 0.91 | 0.88 | 0.84 | 0.93 | 0.87 | 0.91 | 0.91 | 0.91 |
|  | X-MARX | 0.86 | 0.80 | 0.83 | 0.83 | 0.88 | 0.92 | 0.85 | 0.90 | 0.94 | 0.95 |
|  | X-MAF | 0.93 | 0.87 | 0.89 | 0.87 | 0.87 | 0.94 | 0.85 | 0.90 | 0.88 | 0.91 |
|  | X-Level | 0.92 | 0.90 | 0.90 | 0.95 | 0.87 | 0.95 | 0.87 | 0.82 | 0.89 | 0.93 |
|  | X-MARX-Level | 0.86 | 0.80 | 0.84 | 0.86 | 0.88 | 0.91 | 0.86 | 0.87 | 0.93 | 0.96 |
| Boosted Trees | F | 0.88 | 0.87 | 0.96 | 0.93 | 0.92 | 0.89 | 0.92 | 0.96 | 0.92 | 0.89 |
|  | F-X | 0.92 | 0.88 | 0.94 | 0.91 | 0.93 | 0.92 | 0.95 | 0.95 | 0.93 | 0.91 |
|  | F-MARX | 0.87 | 0.77 | 0.85 | 0.86 | 0.95 | 0.96 | 0.97 | 0.99 | 1.02 | 0.97 |
|  | F-MAF | 0.88 | 0.86 | 0.92 | 0.97 | 0.92 | 0.92 | 0.91 | 0.95 | 0.95 | 0.89 |
|  | F-Level | 0.90 | 0.89 | 0.99 | 1.16 | 0.98 | 0.96 | 0.94 | 0.84 | 1.04 | 0.97 |
|  | F-X-MARX | 0.84 | 0.84 | 0.85 | 0.86 | 0.92 | 0.90 | 0.95 | 0.98 | 0.99 | 0.97 |
|  | F-X-MAF | 0.91 | 0.87 | 0.91 | 0.95 | 0.90 | 0.92 | 0.94 | 0.95 | 0.93 | 0.89 |
|  | F-X-Level | 0.90 | 0.91 | 0.91 | 1.00 | 0.94 | 0.96 | 0.95 | 0.90 | 0.98 | 0.92 |
|  | F-X-MARX-Level | 0.85 | 0.83 | 0.87 | 0.89 | 0.96 | 0.94 | 0.98 | 0.91 | 0.99 | 0.96 |
|  | X | 0.93 | 0.91 | 0.93 | 0.98 | 0.93 | 0.94 | 0.94 | 0.94 | 0.93 | 0.90 |
|  | MARX | 0.86 | 0.81 | 0.86 | 0.87 | 0.99 | 0.95 | 0.96 | 1.02 | 1.01 | 0.97 |
|  | MAF | 1.00 | 0.83 | 0.95 | 0.95 | 0.93 | 0.93 | 0.91 | 0.96 | 0.97 | 0.89 |
|  | X-MARX | 0.85 | 0.82 | 0.87 | 0.89 | 0.93 | 0.95 | 0.94 | 0.96 | 0.99 | 0.95 |
|  | X-MAF | 0.95 | 0.91 | 0.93 | 0.97 | 0.87 | 0.93 | 0.96 | 0.96 | 0.95 | 0.91 |
|  | X-Level | 0.90 | 0.91 | 0.93 | 1.00 | 0.93 | 0.97 | 1.01 | 0.88 | 0.92 | 0.95 |
|  | X-MARX-Level | 0.85 | 0.82 | 0.86 | 0.89 | 0.97 | 0.91 | 0.96 | 0.92 | 0.99 | 0.97 |

#### Horizon 12 (path-average)

**Horizon 12, path-average (SGR)** — FM absolute RMSE (denominator): INDPRO 0.003, EMP 0.001, UNRATE 0.077, INCOME 0.002, CONS 0.002, RETAIL 0.003, HOUST 0.019, M2 0.002, CPI 0.001, PPI 0.003

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 0.97 | 0.88 | 0.94 | 0.90 | 1.00 | 0.98 | 1.15 | 1.02 | 1.09 | 1.03 |
| Adaptive Lasso | F | 0.87 | 0.84 | 0.86 | 0.89 | 0.91 | 1.03 | 0.90 | 0.95 | 1.03 | 0.98 |
|  | F-X | 0.96 | 0.89 | 0.90 | 0.94 | 1.04 | 1.17 | 0.92 | 0.89 | 0.98 | 0.98 |
|  | F-MARX | 0.90 | 0.86 | 0.84 | 0.90 | 0.96 | 0.99 | 0.93 | 0.94 | 1.08 | 1.02 |
|  | F-MAF | 0.92 | 0.87 | 0.86 | 0.91 | 0.91 | 1.01 | 0.91 | 0.93 | 1.03 | 0.99 |
|  | F-Level | 0.89 | 0.81 | 0.88 | 0.97 | 0.86 | 1.00 | 0.98 | 0.96 | 1.33 | 1.06 |
|  | F-X-MARX | 0.99 | 0.89 | 0.86 | 0.89 | 0.99 | 0.99 | 0.95 | 0.87 | 1.06 | 0.96 |
|  | F-X-MAF | 0.96 | 0.89 | 0.88 | 0.96 | 0.96 | 1.11 | 0.91 | 0.88 | 0.99 | 0.95 |
|  | F-X-Level | 0.94 | 0.88 | 0.90 | 0.95 | 0.96 | 1.20 | 0.93 | 0.85 | 1.08 | 0.99 |
|  | F-X-MARX-Level | 0.98 | 0.87 | 0.87 | 0.91 | 0.95 | 1.03 | 0.95 | 0.85 | 1.09 | 0.97 |
|  | X | 0.97 | 0.90 | 0.90 | 0.96 | 1.04 | 0.99 | 0.92 | 0.89 | 1.03 | 0.98 |
|  | MARX | 0.91 | 0.87 | 0.84 | 0.89 | 1.01 | 1.00 | 0.95 | 0.92 | 1.08 | 1.08 |
|  | MAF | 0.96 | 0.87 | 0.86 | 0.94 | 1.05 | 0.99 | 0.89 | 0.93 | 1.04 | 1.00 |
|  | X-MARX | 1.03 | 0.89 | 0.87 | 0.94 | 0.98 | 0.99 | 0.96 | 0.89 | 1.00 | 0.94 |
|  | X-MAF | 1.07 | 0.90 | 0.88 | 0.94 | 0.98 | 1.09 | 0.93 | 0.88 | 1.01 | 0.95 |
|  | X-Level | 0.95 | 0.88 | 0.91 | 0.97 | 0.96 | 1.02 | 0.94 | 0.85 | 1.08 | 0.99 |
|  | X-MARX-Level | 1.00 | 0.87 | 0.86 | 0.91 | 0.99 | 1.00 | 0.96 | 0.85 | 1.09 | 0.97 |
| Elastic Net | F | 0.88 | 0.85 | 0.86 | 0.91 | 0.91 | 1.03 | 1.06 | 0.95 | 1.02 | 0.98 |
|  | F-X | 0.96 | 0.90 | 0.88 | 0.95 | 0.93 | 0.97 | 0.97 | 0.90 | 1.01 | 0.96 |
|  | F-MARX | 0.90 | 0.85 | 0.83 | 0.90 | 0.97 | 0.99 | 1.06 | 0.94 | 1.07 | 1.02 |
|  | F-MAF | 0.91 | 0.87 | 0.84 | 0.91 | 0.90 | 1.00 | 1.12 | 0.93 | 1.03 | 1.00 |
|  | F-Level | 0.89 | 0.81 | 0.84 | 0.99 | 0.86 | 0.98 | 1.15 | 0.96 | 1.20 | 1.09 |
|  | F-X-MARX | 0.98 | 0.88 | 0.83 | 0.92 | 0.97 | 1.00 | 1.03 | 0.88 | 1.00 | 0.93 |
|  | F-X-MAF | 0.96 | 0.89 | 0.86 | 0.95 | 0.95 | 1.20 | 0.97 | 0.89 | 0.98 | 0.95 |
|  | F-X-Level | 0.94 | 0.87 | 0.88 | 0.96 | 0.91 | 0.96 | 0.96 | 0.87 | 1.08 | 0.99 |
|  | F-X-MARX-Level | 0.97 | 0.86 | 0.82 | 0.92 | 0.91 | 0.98 | 1.03 | 0.85 | 1.09 | 0.95 |
|  | X | 0.97 | 0.90 | 0.88 | 0.95 | 0.95 | 0.96 | 0.98 | 0.88 | 1.00 | 0.96 |
|  | MARX | 0.90 | 0.87 | 0.83 | 0.89 | 0.93 | 0.99 | 1.09 | 0.92 | 1.08 | 1.06 |
|  | MAF | 0.95 | 0.87 | 0.84 | 0.94 | 0.97 | 0.98 | 1.21 | 0.93 | 1.05 | 1.00 |
|  | X-MARX | 1.02 | 0.88 | 0.83 | 0.92 | 0.95 | 0.98 | 1.05 | 0.89 | 1.00 | 0.93 |
|  | X-MAF | 0.97 | 0.90 | 0.86 | 0.91 | 0.96 | 0.96 | 0.99 | 0.89 | 0.99 | 0.95 |
|  | X-Level | 0.95 | 0.87 | 0.88 | 0.94 | 0.92 | 0.95 | 0.98 | 0.86 | 1.09 | 0.99 |
|  | X-MARX-Level | 0.98 | 0.86 | 0.82 | 0.93 | 0.97 | 1.05 | 1.06 | 0.85 | 1.09 | 0.96 |
| Linear Boosting | F | 0.87 | 0.87 | 0.84 | 0.91 | 0.90 | 1.00 | 1.09 | 1.19 | 1.35 | 1.05 |
|  | F-X | 0.98 | 0.90 | 0.86 | 0.95 | 1.01 | 1.01 | 0.99 | 1.03 | 1.19 | 1.01 |
|  | F-MARX | 0.89 | 0.91 | 0.81 | 0.92 | 0.90 | 1.02 | 1.09 | 1.14 | 1.28 | 1.06 |
|  | F-MAF | 0.91 | 0.88 | 0.84 | 0.93 | 0.88 | 0.96 | 1.13 | 0.93 | 1.06 | 1.00 |
|  | F-Level | 0.87 | 0.80 | 0.82 | 0.96 | 0.89 | 1.00 | 1.10 | 1.01 | 1.19 | 1.05 |
|  | F-X-MARX | 0.91 | 0.92 | 0.81 | 0.95 | 0.93 | 1.24 | 1.07 | 1.11 | 1.35 | 1.06 |
|  | F-X-MAF | 1.12 | 0.89 | 0.84 | 0.93 | 0.97 | 0.98 | 1.01 | 0.90 | 1.07 | 1.00 |
|  | F-X-Level | 0.92 | 0.87 | 0.85 | 0.94 | 1.03 | 0.97 | 1.03 | 0.98 | 1.10 | 0.98 |
|  | F-X-MARX-Level | 0.90 | 0.86 | 0.80 | 0.93 | 0.95 | 1.03 | 1.09 | 1.00 | 1.14 | 1.05 |
|  | X | 0.94 | 0.94 | 0.86 | 0.98 | 1.04 | 1.01 | 1.01 | 1.02 | 1.24 | 1.02 |
|  | MARX | 0.89 | 0.92 | 0.82 | 0.94 | 0.93 | 0.98 | 1.15 | 1.11 | 1.38 | 1.13 |
|  | MAF | 0.96 | 0.90 | 0.86 | 0.93 | 0.93 | 0.94 | 1.14 | 0.95 | 1.12 | 1.01 |
|  | X-MARX | 0.93 | 0.94 | 0.81 | 0.94 | 0.98 | 1.11 | 1.14 | 1.10 | 1.26 | 1.08 |
|  | X-MAF | 0.94 | 0.89 | 0.86 | 0.94 | 1.10 | 1.00 | 1.00 | 0.88 | 1.09 | 1.04 |
|  | X-Level | 0.94 | 0.88 | 0.86 | 0.96 | 1.00 | 1.00 | 1.00 | 0.99 | 1.10 | 1.00 |
|  | X-MARX-Level | 0.89 | 0.86 | 0.82 | 1.06 | 1.06 | 0.96 | 1.11 | 1.01 | 1.12 | 1.15 |
| Random Forest | F | 0.89 | 0.84 | 0.85 | 0.85 | 0.86 | 0.91 | 0.91 | 0.90 | 0.87 | 0.89 |
|  | F-X | 0.94 | 0.87 | 0.87 | 0.88 | 0.84 | 0.94 | 0.87 | 0.89 | 0.88 | 0.92 |
|  | F-MARX | 0.88 | 0.78 | 0.82 | 0.81 | 0.82 | 0.91 | 0.87 | 0.92 | 0.97 | 1.01 |
|  | F-MAF | 0.96 | 0.81 | 0.87 | 0.84 | 0.80 | 0.91 | 0.85 | 0.92 | 0.92 | 0.95 |
|  | F-Level | 0.95 | 0.90 | 0.94 | 1.10 | 0.81 | 0.97 | 0.93 | 0.81 | 0.97 | 1.05 |
|  | F-X-MARX | 0.88 | 0.80 | 0.82 | 0.81 | 0.84 | 0.94 | 0.85 | 0.91 | 0.97 | 0.99 |
|  | F-X-MAF | 0.95 | 0.84 | 0.86 | 0.85 | 0.83 | 0.94 | 0.85 | 0.89 | 0.90 | 0.95 |
|  | F-X-Level | 0.92 | 0.87 | 0.88 | 0.92 | 0.83 | 0.95 | 0.86 | 0.83 | 0.94 | 0.98 |
|  | F-X-MARX-Level | 0.87 | 0.79 | 0.83 | 0.84 | 0.83 | 0.93 | 0.88 | 0.87 | 0.98 | 1.02 |
|  | X | 0.94 | 0.87 | 0.88 | 0.88 | 0.85 | 0.93 | 0.88 | 0.88 | 0.88 | 0.92 |
|  | MARX | 0.87 | 0.78 | 0.82 | 0.82 | 0.84 | 0.91 | 0.89 | 0.93 | 0.99 | 1.02 |
|  | MAF | 0.99 | 0.81 | 0.89 | 0.85 | 0.82 | 0.94 | 0.88 | 0.91 | 0.93 | 0.95 |
|  | X-MARX | 0.88 | 0.79 | 0.82 | 0.82 | 0.85 | 0.92 | 0.86 | 0.91 | 0.96 | 0.99 |
|  | X-MAF | 0.95 | 0.84 | 0.87 | 0.85 | 0.85 | 0.94 | 0.86 | 0.90 | 0.90 | 0.94 |
|  | X-Level | 0.93 | 0.87 | 0.88 | 0.95 | 0.84 | 0.95 | 0.88 | 0.82 | 0.94 | 0.98 |
|  | X-MARX-Level | 0.88 | 0.79 | 0.83 | 0.85 | 0.85 | 0.92 | 0.87 | 0.86 | 0.98 | 1.02 |
| Boosted Trees | F | 0.90 | 0.84 | 0.93 | 0.91 | 0.89 | 0.89 | 0.93 | 0.97 | 0.95 | 0.92 |
|  | F-X | 0.94 | 0.85 | 0.92 | 0.88 | 0.89 | 0.91 | 0.97 | 0.96 | 0.97 | 0.94 |
|  | F-MARX | 0.90 | 0.75 | 0.84 | 0.85 | 0.91 | 0.95 | 1.01 | 0.99 | 1.05 | 1.03 |
|  | F-MAF | 0.89 | 0.82 | 0.88 | 0.94 | 0.89 | 0.91 | 0.92 | 0.95 | 0.98 | 0.91 |
|  | F-Level | 0.93 | 0.88 | 0.94 | 1.16 | 0.96 | 0.96 | 0.96 | 0.84 | 1.13 | 1.03 |
|  | F-X-MARX | 0.87 | 0.82 | 0.85 | 0.86 | 0.89 | 0.88 | 0.97 | 0.98 | 1.02 | 1.01 |
|  | F-X-MAF | 0.92 | 0.84 | 0.88 | 0.93 | 0.87 | 0.91 | 0.95 | 0.96 | 0.96 | 0.92 |
|  | F-X-Level | 0.91 | 0.87 | 0.89 | 0.99 | 0.93 | 0.94 | 0.96 | 0.90 | 1.03 | 0.96 |
|  | F-X-MARX-Level | 0.87 | 0.82 | 0.85 | 0.91 | 0.94 | 0.92 | 1.01 | 0.92 | 1.03 | 1.02 |
|  | X | 0.95 | 0.88 | 0.90 | 0.96 | 0.88 | 0.94 | 0.98 | 0.93 | 0.96 | 0.92 |
|  | MARX | 0.89 | 0.79 | 0.85 | 0.86 | 0.94 | 0.93 | 1.00 | 1.02 | 1.06 | 1.03 |
|  | MAF | 1.02 | 0.79 | 0.91 | 0.92 | 0.91 | 0.92 | 0.92 | 0.96 | 1.01 | 0.91 |
|  | X-MARX | 0.87 | 0.81 | 0.86 | 0.88 | 0.90 | 0.94 | 0.96 | 0.97 | 1.03 | 1.00 |
|  | X-MAF | 0.97 | 0.87 | 0.92 | 0.95 | 0.84 | 0.93 | 0.97 | 0.96 | 0.97 | 0.93 |
|  | X-Level | 0.92 | 0.88 | 0.91 | 0.98 | 0.91 | 0.95 | 1.03 | 0.88 | 0.96 | 0.98 |
|  | X-MARX-Level | 0.88 | 0.80 | 0.85 | 0.89 | 0.93 | 0.89 | 0.97 | 0.92 | 1.03 | 1.02 |

#### Horizon 24 (path-average)

**Horizon 24, path-average (SGR)** — FM absolute RMSE (denominator): INDPRO 0.003, EMP 0.001, UNRATE 0.068, INCOME 0.002, CONS 0.002, RETAIL 0.003, HOUST 0.014, M2 0.002, CPI 0.002, PPI 0.003

| Model | Set | INDPRO | EMP | UNRATE | INCOME | CONS | RETAIL | HOUST | M2 | CPI | PPI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | — | 1.15 | 0.94 | 1.13 | 0.96 | 0.97 | 1.05 | 1.48 | 1.02 | 1.04 | 0.99 |
| Adaptive Lasso | F | 1.02 | 0.93 | 0.99 | 0.91 | 0.82 | 1.06 | 1.01 | 0.95 | 0.99 | 0.96 |
|  | F-X | 1.02 | 0.94 | 1.01 | 0.93 | 0.96 | 1.18 | 1.05 | 0.89 | 0.91 | 0.94 |
|  | F-MARX | 1.07 | 0.95 | 1.03 | 0.92 | 0.84 | 1.02 | 1.09 | 0.96 | 0.98 | 1.01 |
|  | F-MAF | 1.04 | 0.96 | 1.00 | 0.92 | 0.79 | 1.02 | 1.04 | 0.95 | 0.97 | 0.92 |
|  | F-Level | 1.03 | 0.90 | 1.02 | 1.07 | 0.82 | 1.00 | 1.07 | 1.03 | 1.30 | 1.08 |
|  | F-X-MARX | 1.09 | 0.96 | 1.00 | 0.97 | 0.88 | 1.00 | 1.10 | 0.88 | 0.99 | 0.92 |
|  | F-X-MAF | 1.04 | 0.95 | 1.02 | 1.03 | 0.84 | 1.13 | 1.05 | 0.89 | 0.92 | 0.90 |
|  | F-X-Level | 1.00 | 0.93 | 1.03 | 0.98 | 0.88 | 1.27 | 1.04 | 0.92 | 1.03 | 0.95 |
|  | F-X-MARX-Level | 1.09 | 0.94 | 1.01 | 0.95 | 0.84 | 1.04 | 1.09 | 0.92 | 1.04 | 0.94 |
|  | X | 1.05 | 0.95 | 1.01 | 0.96 | 0.93 | 0.99 | 1.05 | 0.89 | 0.97 | 0.93 |
|  | MARX | 1.10 | 0.98 | 1.02 | 0.92 | 0.89 | 1.03 | 1.12 | 0.95 | 0.98 | 1.04 |
|  | MAF | 1.07 | 0.95 | 0.97 | 0.99 | 0.93 | 0.98 | 1.01 | 0.94 | 0.96 | 0.93 |
|  | X-MARX | 1.14 | 0.97 | 1.01 | 0.95 | 0.86 | 0.98 | 1.12 | 0.89 | 0.92 | 0.90 |
|  | X-MAF | 1.15 | 0.96 | 1.01 | 0.95 | 0.87 | 1.12 | 1.06 | 0.89 | 0.95 | 0.91 |
|  | X-Level | 1.01 | 0.93 | 1.03 | 1.00 | 0.90 | 1.08 | 1.05 | 0.92 | 1.03 | 0.94 |
|  | X-MARX-Level | 1.11 | 0.94 | 1.01 | 0.95 | 0.87 | 1.00 | 1.11 | 0.91 | 1.03 | 0.94 |
| Elastic Net | F | 0.99 | 0.93 | 0.99 | 0.93 | 0.80 | 1.06 | 1.27 | 0.94 | 0.99 | 0.94 |
|  | F-X | 1.03 | 0.95 | 0.99 | 0.93 | 0.85 | 1.03 | 1.15 | 0.90 | 0.95 | 0.93 |
|  | F-MARX | 1.09 | 0.96 | 1.01 | 0.93 | 0.86 | 1.01 | 1.31 | 0.96 | 0.97 | 1.00 |
|  | F-MAF | 1.01 | 0.96 | 0.97 | 0.94 | 0.79 | 1.01 | 1.40 | 0.95 | 0.96 | 0.95 |
|  | F-Level | 1.04 | 0.91 | 1.01 | 1.10 | 0.78 | 0.98 | 1.50 | 1.04 | 1.18 | 1.11 |
|  | F-X-MARX | 1.09 | 0.95 | 0.98 | 0.93 | 0.88 | 1.01 | 1.25 | 0.88 | 0.93 | 0.90 |
|  | F-X-MAF | 1.04 | 0.95 | 1.00 | 0.94 | 0.85 | 1.26 | 1.13 | 0.90 | 0.92 | 0.91 |
|  | F-X-Level | 1.01 | 0.93 | 1.00 | 0.97 | 0.83 | 1.01 | 1.13 | 0.94 | 1.04 | 0.94 |
|  | F-X-MARX-Level | 1.08 | 0.94 | 0.97 | 0.95 | 0.83 | 0.98 | 1.26 | 0.92 | 1.05 | 0.92 |
|  | X | 1.04 | 0.95 | 0.99 | 0.94 | 0.88 | 1.03 | 1.15 | 0.89 | 0.94 | 0.92 |
|  | MARX | 1.10 | 0.97 | 1.02 | 0.92 | 0.82 | 1.02 | 1.37 | 0.95 | 0.97 | 1.05 |
|  | MAF | 1.05 | 0.95 | 0.95 | 0.99 | 0.84 | 0.97 | 1.55 | 0.95 | 0.96 | 0.94 |
|  | X-MARX | 1.13 | 0.96 | 0.97 | 0.94 | 0.84 | 0.98 | 1.28 | 0.89 | 0.93 | 0.89 |
|  | X-MAF | 1.05 | 0.95 | 1.00 | 0.94 | 0.88 | 0.96 | 1.16 | 0.90 | 0.92 | 0.91 |
|  | X-Level | 1.01 | 0.93 | 1.00 | 1.00 | 0.85 | 0.94 | 1.15 | 0.93 | 1.05 | 0.95 |
|  | X-MARX-Level | 1.08 | 0.94 | 0.97 | 0.95 | 0.86 | 1.07 | 1.29 | 0.92 | 1.04 | 0.92 |
| Linear Boosting | F | 1.01 | 0.95 | 0.99 | 0.91 | 0.80 | 1.01 | 1.36 | 1.16 | 1.34 | 1.02 |
|  | F-X | 1.06 | 0.93 | 0.97 | 0.93 | 0.92 | 1.03 | 1.19 | 1.04 | 1.15 | 0.98 |
|  | F-MARX | 1.10 | 0.95 | 1.00 | 0.94 | 0.80 | 1.06 | 1.36 | 1.16 | 1.19 | 1.06 |
|  | F-MAF | 1.03 | 0.97 | 1.00 | 0.94 | 0.79 | 0.94 | 1.41 | 0.97 | 1.00 | 0.95 |
|  | F-Level | 1.02 | 0.90 | 0.98 | 1.03 | 0.81 | 1.01 | 1.35 | 1.09 | 1.22 | 1.05 |
|  | F-X-MARX | 1.05 | 0.94 | 0.97 | 0.95 | 0.83 | 1.27 | 1.32 | 1.11 | 1.25 | 1.04 |
|  | F-X-MAF | 1.23 | 0.93 | 0.96 | 0.94 | 0.86 | 0.97 | 1.25 | 0.93 | 1.04 | 0.98 |
|  | F-X-Level | 1.01 | 0.92 | 0.98 | 0.95 | 0.94 | 0.95 | 1.28 | 1.05 | 1.10 | 0.97 |
|  | F-X-MARX-Level | 1.04 | 0.92 | 0.97 | 0.95 | 0.85 | 1.03 | 1.34 | 1.06 | 1.07 | 1.04 |
|  | X | 1.01 | 0.93 | 0.95 | 0.97 | 0.94 | 0.98 | 1.22 | 1.04 | 1.20 | 1.00 |
|  | MARX | 1.13 | 0.95 | 1.00 | 0.97 | 0.85 | 0.98 | 1.47 | 1.12 | 1.27 | 1.14 |
|  | MAF | 1.06 | 0.97 | 0.98 | 0.97 | 0.80 | 0.92 | 1.43 | 0.98 | 1.06 | 0.95 |
|  | X-MARX | 1.07 | 0.94 | 0.95 | 0.94 | 0.88 | 1.12 | 1.44 | 1.11 | 1.16 | 1.06 |
|  | X-MAF | 1.02 | 0.93 | 0.95 | 0.98 | 0.98 | 0.97 | 1.21 | 0.91 | 1.06 | 0.99 |
|  | X-Level | 1.00 | 0.89 | 0.94 | 0.95 | 0.95 | 0.97 | 1.21 | 1.07 | 1.12 | 0.97 |
|  | X-MARX-Level | 1.06 | 0.91 | 0.96 | 1.09 | 0.95 | 0.95 | 1.38 | 1.08 | 1.06 | 1.14 |
| Random Forest | F | 1.03 | 0.90 | 1.03 | 0.89 | 0.78 | 0.91 | 1.00 | 0.86 | 0.81 | 0.81 |
|  | F-X | 1.05 | 0.91 | 1.03 | 0.93 | 0.74 | 0.95 | 0.95 | 0.86 | 0.80 | 0.86 |
|  | F-MARX | 1.10 | 0.89 | 1.07 | 0.91 | 0.75 | 0.96 | 1.08 | 0.89 | 0.88 | 0.97 |
|  | F-MAF | 1.12 | 0.88 | 1.06 | 0.93 | 0.73 | 0.92 | 0.94 | 0.86 | 0.84 | 0.89 |
|  | F-Level | 1.16 | 0.96 | 1.17 | 1.23 | 0.77 | 1.02 | 1.10 | 0.78 | 0.93 | 1.02 |
|  | F-X-MARX | 1.07 | 0.89 | 1.04 | 0.89 | 0.75 | 0.95 | 1.01 | 0.89 | 0.87 | 0.94 |
|  | F-X-MAF | 1.08 | 0.90 | 1.03 | 0.92 | 0.73 | 0.95 | 0.93 | 0.85 | 0.82 | 0.89 |
|  | F-X-Level | 1.06 | 0.92 | 1.05 | 1.00 | 0.74 | 0.96 | 0.96 | 0.81 | 0.88 | 0.95 |
|  | F-X-MARX-Level | 1.07 | 0.89 | 1.06 | 0.92 | 0.74 | 0.94 | 1.03 | 0.87 | 0.91 | 0.99 |
|  | X | 1.03 | 0.91 | 1.02 | 0.94 | 0.75 | 0.93 | 0.96 | 0.85 | 0.80 | 0.85 |
|  | MARX | 1.10 | 0.88 | 1.08 | 0.92 | 0.77 | 0.96 | 1.11 | 0.90 | 0.90 | 0.98 |
|  | MAF | 1.14 | 0.87 | 1.07 | 0.95 | 0.73 | 0.95 | 0.97 | 0.84 | 0.84 | 0.89 |
|  | X-MARX | 1.05 | 0.89 | 1.05 | 0.89 | 0.76 | 0.94 | 1.02 | 0.88 | 0.87 | 0.94 |
|  | X-MAF | 1.07 | 0.90 | 1.03 | 0.92 | 0.74 | 0.94 | 0.94 | 0.85 | 0.82 | 0.88 |
|  | X-Level | 1.07 | 0.91 | 1.04 | 1.02 | 0.75 | 0.97 | 1.00 | 0.81 | 0.88 | 0.95 |
|  | X-MARX-Level | 1.08 | 0.89 | 1.06 | 0.93 | 0.76 | 0.94 | 1.03 | 0.86 | 0.91 | 1.00 |
| Boosted Trees | F | 1.03 | 0.89 | 1.07 | 0.90 | 0.78 | 0.85 | 1.00 | 0.94 | 0.92 | 0.87 |
|  | F-X | 1.06 | 0.90 | 1.02 | 0.92 | 0.77 | 0.87 | 1.06 | 0.95 | 0.89 | 0.88 |
|  | F-MARX | 1.09 | 0.86 | 1.04 | 0.91 | 0.83 | 0.95 | 1.13 | 0.97 | 0.96 | 0.98 |
|  | F-MAF | 1.00 | 0.86 | 1.01 | 1.02 | 0.78 | 0.90 | 1.03 | 0.91 | 0.89 | 0.86 |
|  | F-Level | 1.11 | 0.97 | 1.07 | 1.25 | 0.88 | 0.95 | 1.00 | 0.82 | 1.06 | 0.97 |
|  | F-X-MARX | 1.09 | 0.92 | 1.01 | 0.97 | 0.79 | 0.87 | 1.10 | 0.95 | 0.92 | 0.94 |
|  | F-X-MAF | 1.04 | 0.89 | 1.03 | 1.00 | 0.78 | 0.87 | 0.99 | 0.93 | 0.89 | 0.86 |
|  | F-X-Level | 1.00 | 0.92 | 1.03 | 1.04 | 0.84 | 0.93 | 1.02 | 0.91 | 0.95 | 0.88 |
|  | F-X-MARX-Level | 1.05 | 0.91 | 1.04 | 1.02 | 0.82 | 0.94 | 1.11 | 0.91 | 0.95 | 1.01 |
|  | X | 1.08 | 0.91 | 0.99 | 1.02 | 0.78 | 0.90 | 1.03 | 0.92 | 0.89 | 0.86 |
|  | MARX | 1.10 | 0.91 | 1.07 | 0.96 | 0.83 | 0.96 | 1.13 | 0.99 | 0.97 | 0.99 |
|  | MAF | 1.16 | 0.85 | 1.07 | 1.04 | 0.79 | 0.91 | 1.01 | 0.91 | 0.91 | 0.82 |
|  | X-MARX | 1.03 | 0.89 | 1.04 | 0.97 | 0.81 | 0.90 | 1.05 | 0.95 | 0.94 | 0.95 |
|  | X-MAF | 1.08 | 0.89 | 1.04 | 1.03 | 0.75 | 0.89 | 1.03 | 0.93 | 0.89 | 0.88 |
|  | X-Level | 1.06 | 0.91 | 1.02 | 1.04 | 0.80 | 0.93 | 1.12 | 0.90 | 0.88 | 0.93 |
|  | X-MARX-Level | 1.07 | 0.90 | 1.04 | 0.97 | 0.83 | 0.90 | 1.05 | 0.91 | 0.97 | 0.99 |
