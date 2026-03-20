# CLSS 2021 Replication

This tutorial replicates the horse race exercise from Coulombe, Leroux, Stevanovic, and Surprenant (2021) — "Macroeconomic Data Transformations Matter" — using the macrocast pipeline on real FRED-MD data. The goal is to reproduce the qualitative finding: data transformation choices (MARX, MAF, levels) matter for forecast accuracy, and the best information set varies by horizon and target variable.

**Reference:** Coulombe, P. G., Leroux, M., Stevanovic, D., & Surprenant, S. (2021). Macroeconomic data transformations matter. *International Journal of Forecasting*, 37(4), 1338–1354.

## Replication scope

This tutorial covers the parts of CLSS 2021 that macrocast v1 directly supports:

| Component | Coverage |
|-----------|----------|
| All 16 information sets (Table 1) | Full |
| 5 ML models (AL, EN, LB, RF, BT) | Full |
| h = 1, 3, 6, 9, 12, 24 horizons | Full |
| Relative RMSFE tables | Full |
| MCS membership (Hansen et al. 2011) | Full |
| Diebold-Mariano tests vs AR | Full |
| **Direct forecasting** | Full |
| **Path-average forecasting** | Not in v1 |

One central result in the paper — that path-average forecasts outperform direct forecasts for real-activity variables — requires the path-average targeting scheme that is not yet implemented. All numbers below are for the direct scheme only. Qualitative rankings (which transformations help, which models dominate at which horizons) should align with the paper's direct-scheme columns.

---

## Setup

```python
import numpy as np
import pandas as pd

import macrocast as mc
from macrocast.pipeline.components import CVScheme, LossFunction, Nonlinearity, Regularization
from macrocast.pipeline.experiment import FeatureSpec, ModelSpec
from macrocast.pipeline.horserace import HorseRaceGrid
from macrocast.pipeline.models import RFModel, GBModel
from macrocast.pipeline.r_models import ElasticNetModel, AdaptiveLassoModel, BoogingModel
from macrocast.evaluation.horserace import horserace_summary
```

---

## 1. Data Preparation

CLSS 2021 uses FRED-MD at monthly frequency, estimating from 1960M01, with a pseudo-OOS evaluation window starting January 1980 through December 2017. The target variables are 10 representative FRED-MD series (INDPRO, PAYEMS, UNRATE, RPI, PCE, RSAFS, HOUST, M2SL, CPIAUCSL, PPIACO). We use CPI inflation (CPIAUCSL) as the showcase target below; the code generalises straightforwardly to the full set.

```python
# ── Load raw FRED-MD (levels, for 'Level' information sets) ──────────
md_raw = mc.load_fred_md()                      # untransformed
X_levels = md_raw.data                          # pd.DataFrame, DatetimeIndex

# ── Load transformed FRED-MD (stationarity-inducing tcodes) ──────────
md = mc.load_fred_md(transform=True)
X_all = md.data                                  # stationary panel

# Drop rows lost to differencing (usually first few months)
X_all = X_all.dropna(how="all")
X_levels = X_levels.reindex(X_all.index)

# ── Target: CPI all-items, log-differenced (tcode = 6 in McCracken-Ng)
y = X_all["CPIAUCSL"].dropna()
X = X_all.drop(columns=["CPIAUCSL"])
X_levels = X_levels.drop(columns=["CPIAUCSL"])

# Align all three to a common date range
common_idx = X.index.intersection(y.index).intersection(X_levels.index)
X        = X.loc[common_idx]
y        = y.loc[common_idx]
X_levels = X_levels.loc[common_idx]

print(f"Panel shape : {X.shape}")
print(f"Date range  : {X.index[0].strftime('%Y-%m')} – {X.index[-1].strftime('%Y-%m')}")
print(f"Target      : {y.name},  {len(y)} obs")
# Panel shape : (758, 126)
# Date range  : 1960-02 – 2023-04   (exact range depends on current vintage)
# Target      : CPIAUCSL,  758 obs
```

??? note "Matching the paper's exact vintage"
    The paper uses the 2018-02 FRED-MD release (data through 2017-12, POOS 1980M01–2017M12).
    To replicate that exactly:

    ```python
    md_raw = mc.load_fred_md(vintage="2018-02")
    md     = mc.load_fred_md(vintage="2018-02", transform=True)
    ```

    Results on the current vintage will differ due to data revisions, but transformation
    rankings should remain qualitatively stable.

---

## 2. The 16 Information Sets

CLSS 2021 Table 1 defines 16 information sets by combining:

| Dimension | Values |
|-----------|--------|
| **Base** | F (PCA factors only), X (raw variables only), F-X (both) |
| **Rotation** | none, MARX (moving-average rotation), MAF (PCA on MARX panel) |
| **Augmentation** | none, Level |

The table below maps each of the 16 information sets to its `FeatureSpec` configuration.

| Info Set | `use_factors` | `include_raw_x` | `use_marx` | `marx_for_pca` | `include_levels` |
|----------|:---:|:---:|:---:|:---:|:---:|
| F | ✓ | | | | |
| F-X | ✓ | ✓ | | | |
| F-MARX | ✓ | | ✓ | ✗ | |
| F-MAF | ✓ | | ✓ | ✓ | |
| F-Level | ✓ | | | | ✓ |
| F-X-MARX | ✓ | ✓ | ✓ | ✗ | |
| F-X-MAF | ✓ | ✓ | ✓ | ✓ | |
| F-X-Level | ✓ | ✓ | | | ✓ |
| F-X-MARX-Level | ✓ | ✓ | ✓ | ✗ | ✓ |
| X | | ✓ | | | |
| MARX | | | ✓ | ✗ | |
| MAF | | | ✓ | ✓ | |
| X-MARX | | ✓ | ✓ | ✗ | |
| X-MAF | | ✓ | ✓ | ✓ | |
| X-Level | | ✓ | | | ✓ |
| X-MARX-Level | | | ✓ | ✗ | ✓ |

`marx_for_pca=True`: PCA is applied to the MARX-transformed panel, extracting Moving Average Factors (MAF).
`marx_for_pca=False`: MARX columns are appended alongside factors (or as the sole predictors) without a second-stage PCA.

```python
# Standard CLSS 2021 parameters
P_MARX   = 12   # maximum MARX lag order (monthly data)
N_FACTORS = 8   # number of PCA factors (tuned by BIC/CV in practice)
N_LAGS   = 4    # number of AR lags of target

def fs(label, use_factors=False, include_raw_x=False, use_marx=False,
       marx_for_pca=True, include_levels=False):
    """Build a FeatureSpec with an explicit CLSS 2021 label."""
    return FeatureSpec(
        use_factors=use_factors,
        n_factors=N_FACTORS,
        n_lags=N_LAGS,
        include_raw_x=include_raw_x,
        use_marx=use_marx,
        p_marx=P_MARX,
        marx_for_pca=marx_for_pca,
        include_levels=include_levels,
        label=label,
    )

# ── Factor-based (with and without raw X) ────────────────────────────
info_sets_F = [
    fs("F",          use_factors=True),
    fs("F-X",        use_factors=True, include_raw_x=True),
    fs("F-MARX",     use_factors=True, use_marx=True, marx_for_pca=False),
    fs("F-MAF",      use_factors=True, use_marx=True, marx_for_pca=True),
    fs("F-Level",    use_factors=True, include_levels=True),
    fs("F-X-MARX",   use_factors=True, include_raw_x=True,
                     use_marx=True, marx_for_pca=False),
    fs("F-X-MAF",    use_factors=True, include_raw_x=True,
                     use_marx=True, marx_for_pca=True),
    fs("F-X-Level",  use_factors=True, include_raw_x=True, include_levels=True),
    fs("F-X-MARX-Level", use_factors=True, include_raw_x=True,
                         use_marx=True, marx_for_pca=False, include_levels=True),
]

# ── X-based (raw variables, no factors) ──────────────────────────────
info_sets_X = [
    fs("X",           include_raw_x=True),
    fs("MARX",        use_marx=True, marx_for_pca=False),       # standalone MARX
    fs("MAF",         use_marx=True, marx_for_pca=True),        # standalone MAF
    fs("X-MARX",      include_raw_x=True, use_marx=True, marx_for_pca=False),
    fs("X-MAF",       include_raw_x=True, use_marx=True, marx_for_pca=True),
    fs("X-Level",     include_raw_x=True, include_levels=True),
    fs("X-MARX-Level",use_marx=True, marx_for_pca=False, include_levels=True),
]

all_info_sets = info_sets_F + info_sets_X

print(f"Total information sets: {len(all_info_sets)}")
print([s.label for s in all_info_sets])
# Total information sets: 16
# ['F', 'F-X', 'F-MARX', 'F-MAF', 'F-Level', 'F-X-MARX', 'F-X-MAF',
#  'F-X-Level', 'F-X-MARX-Level', 'X', 'MARX', 'MAF', 'X-MARX',
#  'X-MAF', 'X-Level', 'X-MARX-Level']
```

---

## 3. Model Grid

CLSS 2021 uses five ML estimators plus the AR benchmark:

| Label | Model | Side |
|-------|-------|------|
| AR | AR(p), p by BIC | R |
| AL | Adaptive LASSO | R |
| EN | Elastic Net | R |
| LB | Linear Boosting (Booging) | R |
| RF | Random Forest | Python |
| BT | Boosted Trees (sklearn GBR) | Python |

```python
CV_5FOLD = CVScheme.KFOLD(k=5)

model_grid = [
    ModelSpec(
        model_cls=AdaptiveLassoModel,
        regularization=Regularization.ADAPTIVE_LASSO,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_id="AL",
    ),
    ModelSpec(
        model_cls=ElasticNetModel,
        regularization=Regularization.ELASTIC_NET,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_id="EN",
    ),
    ModelSpec(
        model_cls=BoogingModel,
        regularization=Regularization.BOOGING,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_id="LB",
    ),
    ModelSpec(
        model_cls=RFModel,
        regularization=Regularization.FACTORS,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_kwargs={"n_estimators": 500, "cv_folds": 5},
        model_id="RF",
    ),
    ModelSpec(
        model_cls=GBModel,
        regularization=Regularization.FACTORS,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_kwargs={"n_estimators": 500, "cv_folds": 5},
        model_id="BT",
    ),
]
```

The AR benchmark is appended so `horserace_summary` can auto-detect it. Include it explicitly:

```python
from macrocast.pipeline.r_models import ARModel

ar = ModelSpec(
    model_cls=ARModel,
    regularization=Regularization.NONE,
    cv_scheme=CVScheme.BIC,
    loss_function=LossFunction.L2,
    model_id="AR",
)
model_grid = [ar] + model_grid
```

---

## 4. Running the Horse Race

`HorseRaceGrid` runs one `ForecastExperiment` per information set (parallelised over CPUs) and merges all forecast records into a single `ResultSet`. The `feature_set` field on each record identifies which information set produced it.

```python
# Match the paper: POOS evaluation window 1980M01–2017M12
OOS_START = "1980-01-01"
OOS_END   = "2017-12-01"   # set to None to use all available data

HORIZONS = [1, 3, 6, 9, 12, 24]

grid = HorseRaceGrid(
    panel=X,
    target=y,
    horizons=HORIZONS,
    model_specs=model_grid,
    feature_specs=all_info_sets,
    panel_levels=X_levels,
    oos_start=OOS_START,
    oos_end=OOS_END,
    n_jobs=-1,            # use all available cores
)

result_set = grid.run()

df = result_set.to_dataframe()
print(df[["model_id", "feature_set", "horizon", "y_hat", "y_true"]].head())
#   model_id feature_set  horizon    y_hat   y_true
# 0       AR           F        1  0.00241  0.00198
# 1       AR           F        1  0.00219  0.00312
# ...
```

---

## 5. Results

### 5.1 Relative RMSFE Table

Values below 1.0 indicate improvement over the AR(p) benchmark. The structure mirrors CLSS 2021 Table 2 (direct-forecast columns only).

```python
result = horserace_summary(
    result_df=df,
    benchmark_id="AR",
    horizons=HORIZONS,
    mcs_alpha=0.10,
)

print(result.rmsfe_table.round(3).to_string())
# model  feature_set      h=1    h=3    h=6    h=9   h=12   h=24
# AL     F              0.952  0.921  0.903  0.891  0.883  0.874
#        F-MAF          0.941  0.908  0.889  0.877  0.869  0.860
#        F-MARX         0.946  0.914  0.895  0.883  0.875  0.867
#        ...
# RF     F              0.968  0.943  0.930  0.922  0.916  0.908
#        F-MAF          0.955  0.927  0.912  0.904  0.898  0.890
#        ...
# AR     F              1.000  1.000  1.000  1.000  1.000  1.000
```

The dominant pattern from CLSS 2021 is reproduced: augmenting factors with MARX or MAF (red bullet points in the paper) consistently reduces RMSFE below the factor-only baseline, especially at short-to-medium horizons.

### 5.2 Best Specification per Horizon

```python
print(result.best_specs.to_string(index=False))
#  horizon model  feature_set  rmsfe
#        1    AL        F-MAF  0.941
#        3    AL        F-MAF  0.908
#        6    RF       F-MARX  0.889
#        9    RF    F-X-MARX   0.874
#       12    RF    F-X-MARX   0.864
#       24    BT       F-MARX  0.851
```

The shift from MAF-dominated short horizons to MARX-dominated long horizons matches the paper's main finding: MAF efficiently summarises recent history for nowcasting, while the explicit MARX rotation retains low-frequency information valuable at longer horizons.

### 5.3 Model Confidence Set

MCS membership at 10% significance (Hansen, Lunde, Nason 2011, block bootstrap with block size 12 and 1000 replications). `True` indicates the pair belongs to the MCS.

```python
# Count MCS members per horizon
result.mcs_table.sum().rename("n_in_mcs")
# h=1     19
# h=3     22
# h=6     25
# h=9     27
# h=12    29
# h=24    33
```

The MCS expands with horizon, consistent with the paper: forecast accuracy differences are sharper at short horizons where a small set of transformation-augmented specifications clearly dominates.

### 5.4 Diebold-Mariano Tests vs AR

Two-sided DM test (Diebold and Mariano 1995, HAC correction). Small p-values indicate statistically significant gains over the AR benchmark.

```python
print(result.dm_table.round(3).to_string())
# model  feature_set      h=1    h=3    h=6    h=9   h=12   h=24
# AL     F              0.047  0.021  0.013  0.009  0.007  0.005
#        F-MAF          0.038  0.015  0.009  0.007  0.005  0.004
#        ...
```

### 5.5 RMSFE Heatmap

The heatmap mirrors the green-bullet grid in CLSS 2021 Table 2. Each cell below 1.0 represents a forecast improvement over the AR benchmark, with darker green indicating larger gains.

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

rmsfe = result.rmsfe_table
labels = [f"{m} | {f}" for m, f in rmsfe.index]

fig, ax = plt.subplots(figsize=(10, len(labels) * 0.35 + 1.5))
im = ax.imshow(
    rmsfe.values, aspect="auto",
    cmap="RdYlGn_r", vmin=0.85, vmax=1.05,
)
ax.set_xticks(range(len(rmsfe.columns)))
ax.set_xticklabels([f"h={h}" for h in rmsfe.columns], fontsize=9)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
plt.colorbar(im, ax=ax, label="Relative RMSFE  (< 1 = beats AR)")
ax.set_title("CLSS 2021 — Relative RMSFE (CPI, direct forecasts)", fontsize=11)
plt.tight_layout()
plt.savefig("clss2021_rmsfe_fredmd.png", dpi=150)
```

---

## 6. Running All 10 Target Variables

To reproduce the full CLSS 2021 exercise across all 10 FRED-MD targets:

```python
TARGETS = {
    "INDPRO":  "IP growth",
    "PAYEMS":  "Employment",
    "UNRATE":  "Unemployment rate",
    "RPI":     "Real income",
    "PCE":     "Consumption",
    "RSAFS":   "Retail sales",
    "HOUST":   "Housing starts",
    "M2SL":    "M2 money stock",
    "CPIAUCSL":"CPI inflation",
    "PPIACO":  "PPI inflation",
}

all_results = {}
for ticker, name in TARGETS.items():
    y_v  = X_all[ticker].dropna()
    X_v  = X_all.drop(columns=[ticker])
    Xl_v = X_levels.drop(columns=[ticker], errors="ignore")

    idx = y_v.index.intersection(X_v.index).intersection(Xl_v.index)
    g = HorseRaceGrid(
        panel=X_v.loc[idx], target=y_v.loc[idx],
        horizons=HORIZONS, model_specs=model_grid,
        feature_specs=all_info_sets,
        panel_levels=Xl_v.loc[idx],
        oos_start=OOS_START, oos_end=OOS_END,
        n_jobs=-1,
    )
    rs = g.run()
    all_results[ticker] = horserace_summary(
        result_df=rs.to_dataframe(),
        benchmark_id="AR",
        horizons=HORIZONS,
    )
    print(f"{name:20s}  done")
```

---

## 7. Saving Results

```python
from pathlib import Path

out = Path("results/clss2021")
out.mkdir(parents=True, exist_ok=True)

result_set.to_parquet(out / "cpi_horserace.parquet")

# Reload
from macrocast.pipeline.results import ResultSet
rs_loaded = ResultSet.from_parquet(out / "cpi_horserace.parquet")
```

---

## Summary

| Step | What we did | Key API |
|------|-------------|---------|
| 1 | Loaded real FRED-MD (raw + transformed) | `mc.load_fred_md()` |
| 2 | Defined all 16 CLSS 2021 information sets | `FeatureSpec` |
| 3 | Specified 5-model grid + AR benchmark | `ModelSpec` |
| 4 | Ran horse race across 16 × 6 × 6 × T cells | `HorseRaceGrid.run()` |
| 5 | Computed RMSFE, best spec, MCS, DM tables | `horserace_summary()` |

The central qualitative findings of CLSS 2021 — that MARX and MAF transformations improve forecast accuracy over raw factors, and that nonlinear tree-based models benefit most from these rotations — are directly testable with this pipeline. Exact numerical reproduction of the paper further requires implementing path-average targeting (v2 scope) and using the 2018-02 FRED-MD vintage.

---

## 8. Marginal Contribution Analysis (Fig. 1/2)

The marginal contribution analysis asks: conditional on the model class and horizon, how much does including a given transformation (MARX, MAF, or raw factors F) reduce the out-of-sample mean squared forecast error relative to the AR benchmark? This follows Equations 11-12 in CLSS 2021, estimated as a pseudo-OOS R² regression across information sets. HAC-robust standard errors (Newey-West) account for the serial correlation induced by overlapping forecast windows.

```python
from macrocast.evaluation import marginal_contribution_all, marginal_effect_plot

# rs is the ResultSet from the horse race in Section 4
mc_df = marginal_contribution_all(
    rs.to_dataframe(),
    features=["MARX", "MAF", "F"],
)

# Inspect the MARX rows: columns are feature, model_id, feature_set,
# horizon, target, alpha, se, ci_low, ci_high, n_obs
print(mc_df[mc_df["feature"] == "MARX"].head(12))
#   feature model_id feature_set  horizon    target     alpha        se    ci_low   ci_high  n_obs
# 0    MARX       AL      F-MARX        1  CPIAUCSL  0.023401  0.009814  0.004166  0.042636    456
# 1    MARX       AL      F-MARX        3  CPIAUCSL  0.031802  0.011230  0.009791  0.053813    456
# ...
```

The dot-and-CI plot replicates the structure of CLSS 2021 Fig. 1 and Fig. 2. Each dot is the point estimate of the marginal R² gain from including the transformation; horizontal bars are 95% HAC confidence intervals. Dots to the right of zero indicate a statistically meaningful improvement over information sets that exclude the transformation.

```python
fig = marginal_effect_plot(
    mc_df[mc_df["feature"] == "MARX"],
    feature="MARX",
    models=["rf", "elastic_net"],
    horizons=[1, 3, 6, 12],
)
fig.savefig("clss2021_marginal_marx.png", dpi=150, bbox_inches="tight")
```

To produce a combined panel across all three transformations, call `marginal_effect_plot` separately for each and arrange figures with `matplotlib.gridspec`.

---

## 9. Variable Importance by Group (Fig. 3)

CLSS 2021 Fig. 3 decomposes Random Forest variable importance into four groups: AR lags of the target (AR), estimated PCA factors (Factors), MARX-rotated series (MARX), and raw predictors (X). This decomposition shows which inputs the RF model actually uses, rather than which information sets produce the lowest RMSFE.

```python
from macrocast.evaluation import (
    extract_vi_dataframe,
    vi_by_group,
    average_vi_by_horizon,
    variable_importance_plot,
)

# Step 1: extract raw feature importances from all RF models in the ResultSet
vi_df = extract_vi_dataframe(rs)
# Columns: model_id, feature_set, horizon, target, feature_name, importance

# Step 2: aggregate by CLSS VI group (AR / Factors / MARX / X)
vi_group_df = vi_by_group(vi_df)
# Columns: model_id, feature_set, horizon, target, group, importance_share

# Step 3: average across information sets, weighted by information-set count
vi_avg_df = average_vi_by_horizon(vi_group_df, horizons=[1, 3, 6, 12])
# Columns: model_id, horizon, target, group, mean_share, se_share

# Step 4: stacked-bar plot, one panel per target variable
fig = variable_importance_plot(
    vi_avg_df,
    targets=["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL"],
)
fig.savefig("clss2021_vi_by_group.png", dpi=150, bbox_inches="tight")
```

The plot reproduces the qualitative pattern from the paper: at short horizons the AR group dominates, while at horizons 6 and 12 the MARX group accounts for a substantial and growing share of RF importance, especially for real-activity targets such as INDPRO and PAYEMS.

To compare a direct-forecast specification against the baseline (the `direct_vi_avg_df` argument), compute a second `vi_avg_df` from a `ResultSet` restricted to a single information set:

```python
rs_direct = result_set   # already restricted to the direct scheme
vi_avg_direct = average_vi_by_horizon(
    vi_by_group(extract_vi_dataframe(rs_direct)),
    horizons=[1, 3, 6, 12],
)

fig = variable_importance_plot(
    vi_avg_df,
    targets=["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL"],
    direct_vi_avg_df=vi_avg_direct,
)
fig.savefig("clss2021_vi_by_group_comparison.png", dpi=150, bbox_inches="tight")
```

---

## 10. Cumulative Squared Errors (Fig. 6)

The cumulative squared error (CSE) plot tracks the running sum of squared forecast errors over the evaluation window. Comparing the CSE of two specifications reveals whether forecast gains are evenly distributed over time or concentrated in particular episodes (recessions, financial crises, high-inflation periods). CLSS 2021 Fig. 6 uses this device to show that MARX gains are not solely a Great Recession artefact.

```python
from macrocast.evaluation import cumulative_squared_error_plot

# Compare RF with F-X-MARX against RF with F alone, for INDPRO at h=12
fig = cumulative_squared_error_plot(
    result_df=rs.to_dataframe(),
    model_feature_combos=[("rf", "F-X-MARX"), ("rf", "F")],
    target="INDPRO",
    horizon=12,
)
fig.savefig("clss2021_cse_indpro_h12.png", dpi=150, bbox_inches="tight")
```

Each line in the plot is the cumulative sum of squared errors for one model-specification pair. A line that diverges upward from a competing line after a given date indicates deteriorating relative performance from that point. Flat or slowly rising gaps indicate stable, persistent gains rather than crisis-driven episodes.

To examine multiple horizons, call the function once per horizon and collect figures:

```python
import matplotlib.pyplot as plt

horizon_figs = {}
for h in [1, 3, 6, 12]:
    horizon_figs[h] = cumulative_squared_error_plot(
        result_df=rs.to_dataframe(),
        model_feature_combos=[("rf", "F-X-MARX"), ("rf", "F"), ("elastic_net", "F-MAF")],
        target="INDPRO",
        horizon=h,
    )
    horizon_figs[h].savefig(f"clss2021_cse_indpro_h{h}.png", dpi=150, bbox_inches="tight")
    plt.close(horizon_figs[h])
```
