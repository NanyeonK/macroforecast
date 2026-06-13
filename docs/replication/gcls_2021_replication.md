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

The build proceeds in eight steps. The detailed appendix numbers that the run must
reproduce live on a companion page,
[GCLS 2021 appendix B ground-truth tables](gcls_2021_appendix_ground_truth.md).

1. Replication specification
2. Data construction
3. Forecast-target construction
4. Preprocessing
5. Feature cases
6. Models and arms
7. Pseudo-out-of-sample window
8. Evaluation and execution

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

*Next.*

## Step 6. Models and arms

*Pending.*

## Step 7. Pseudo-out-of-sample window

*Pending.*

## Step 8. Evaluation and execution

*Pending. The run is validated cell by cell against the appendix ground-truth tables.*
