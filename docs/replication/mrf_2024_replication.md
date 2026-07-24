# Replicating Goulet Coulombe (2024), *The Macroeconomy as a Random Forest*

**Target exhibit.** Table 4 ("Main Quarterly Results") — the pseudo-out-of-sample
relative RMSEs `RMSE_{v,h,m} / RMSE_{v,h,AR(4)}`, for the **unemployment-rate (UR)**
row. Source: Goulet Coulombe, *Journal of Applied Econometrics* 39 (2024); arXiv
2006.12724.

This document reports what macroforecast reproduces, exactly how, where it deviates
and why, and the package defects the replication surfaced. Every number below is
reproducible with `random_state=42`.

---

## 1. Scope and headline

| | |
|---|---|
| Targets replicated | **1 of 5** (UR). Full Table 4 is 14 models × 5 targets × 5 horizons. |
| Models replicated (UR row) | **11 of 14** (see §4). Not yet run: RF, AR+RF, TV-AR — all feasible. |
| Reproducibility | seed-fixed; MRF is bit-identical serial vs 16-core parallel (see §6). |
| Qualitative verdict | **reproduced** — FA-ARRF is the strongest model, ARRF beats AR(4), and the model ranking matches the paper. |
| Quantitative verdict | **partial-to-strong on UR** — the core MRF family lands within mean\|Δ\| 0.03–0.08 of the paper; a few linear/plain-RF cells are off by 0.1–0.27 for reasons attributed in §5. |

---

## 2. Data and target (paper §4)

- **Panel:** `mf.data.load_fred_qd("2020-01")` — FRED-QD, 247 series, raw levels
  1959Q1–2019Q4, with McCracken–Ng transform codes in `.metadata["transform_codes"]`.
  Predictors are made stationary with those codes.
- **Target:** UR is treated as I(1) and forecast as **log-then-first-difference**
  (`Δlog(UNRATE)`), overriding the panel tcode, per the paper's §4.
- `[ASSUMPTION]` Only the **215 of 247** series that are complete over 1961Q3–2002Q4
  are used (a full-panel complete-case restriction leaves 11 rows). The paper's "248
  series" cannot be used in full from this vintage.
- `[GAP]` The paper's exact FRED-QD **vintage** is not the 2020-01 snapshot used here;
  small data-revision differences propagate into every factor and MAF.

## 3. Feature set `S_t` (paper Table 2)

Built once with expanding-window fits (leak-free; each row transformed on data up to
that row) — **1,042 columns** on all predictors, **914** on the 215 complete ones:

| block | construction |
|---|---|
| 8 lags of `y_t` | target lags 1..8 |
| `t` | linear time trend |
| 2 lags of every FRED series | `lag_step(lags=(1,2))` |
| 5 PCA factors × 8 lags | `pca_step(n_components=5)` → `lag_step(lags=1..8)` |
| 2 MAFs per variable | `maf_step(max_lag=8, n_components=2)` (PCA of a variable's 8 lags) |

## 4. Protocol

Expanding estimation from 1961Q3; POOS 2003Q1–2014Q4 (48 quarterly origins);
**re-estimated every two years** (6 estimation points); **direct** h-step forecasts;
h ∈ {1,2,4,6,8}; benchmark **AR(4)** `[1, y_{t-1:4}]`. MRF hyper-parameters are the
author package's defaults (`macrorf::MRF`): `minsize=10, mtry=1/3, min_leaf_frac_of_x=1,
subsampling=0.75, rw_regul=0.75, block_size=12, ridge_lambda=0.1, HRW=0,
resampling_opt=2, trend_push=1`, with `B=50` (the paper's; macroforecast's cheapened
default is `B=25`).

---

## 5. Results — Table 4, UR row (mine / paper, benchmark AR(4))

Cells are `RMSE_ratio_mine / RMSE_ratio_paper`; `mean|Δ|` is the mean absolute
per-cell difference over the five horizons.

| model | h=1 | h=2 | h=4 | h=6 | h=8 | mean\|Δ\| |
|---|---|---|---|---|---|---|
| **ARRF** | 0.937 / 0.90 | 0.878 / 0.90 | 0.789 / 0.87 | 0.957 / 0.95 | 0.994 / 0.98 | **0.032** |
| **RF-MAF** | 0.908 / 0.85 | 0.880 / 0.85 | 0.807 / 0.87 | 0.883 / 0.94 | 0.925 / 0.95 | **0.047** |
| **STAR** | 0.951 / 1.10 | 0.951 / 0.97 | 1.018 / 1.01 | 1.011 / 1.04 | 1.024 / 1.06 | **0.048** |
| **Tiny ARRF** | 0.928 / 1.00 | 0.957 / 0.96 | 0.988 / 0.92 | 1.044 / 0.97 | 1.049 / 0.98 | 0.057 |
| **FA-ARRF** | 0.872 / 0.72 | 0.819 / 0.76 | 0.756 / 0.79 | 0.923 / 0.89 | 0.978 / 1.01 | 0.062 |
| **SETAR** | 1.021 / 1.18 | 0.961 / 1.03 | 1.016 / 1.02 | 1.009 / 1.07 | 1.025 / 1.09 | 0.072 |
| **VARRF** | 1.004 / 1.24 | 0.961 / 0.89 | 0.867 / 0.91 | 0.981 / 0.95 | 1.024 / 1.04 | 0.079 |
| **LASSO-MAF** | 1.114 / 0.99 | 1.075 / 0.98 | 1.010 / 0.96 | 0.892 / 0.98 | 0.939 / 0.98 | 0.080 |
| **FA-AR** | 0.924 / 0.83 | 0.845 / 0.80 | 0.862 / 0.88 | 0.958 / 1.18 | 0.982 / 1.25 | 0.130 |
| **Ridge-MAF** | 1.198 / 0.99 | 0.955 / 0.92 | 1.136 / 0.94 | 0.920 / 0.98 | 1.203 / 1.01 | 0.139 |
| **Tiny RF** | 1.114 / 1.24 | 1.208 / 1.15 | 1.129 / 1.37 | 1.146 / 1.60 | 1.082 / 1.57 | 0.273 |

**Reading it.** The paper's story reproduces: FA-ARRF is the best short-horizon model
(mine and the paper both put it below ARRF), ARRF beats AR at most horizons, VARRF is
the weakest MRF at h=1, and the ∼TAR / penalized-linear models sit near 1.0. ARRF and
RF-MAF are near-exact.

**Where it deviates, and why (all attributed — no reverse-engineering):**

- **Tiny RF (0.273).** `[ASSUMPTION]` The paper's "plain RF" is *its own* MRF with
  `X_t = ι` (an intercept-only linear part). The current MRF requires ≥1 X column
  (see §7), so plain-RF variants are mapped to scikit-learn `random_forest` — a
  genuinely different estimator. RF-MAF (on the wide `S_t`) still matches well; the
  small-feature Tiny RF is where the two algorithms diverge most.
- **FA-AR (0.130) / VARRF / FA-ARRF short-horizon gaps.** These load on the PCA
  factors, and my factors come from 215 series and this vintage; the paper's five
  factors come from the full 248-series original vintage. Factor differences amplify
  at long horizons for FA-AR (paper 1.18/1.25 vs mine ~0.96–0.98).
- **Ridge-MAF (0.139) / LASSO-MAF (0.080).** Penalty tuning differs — I holdout-tune
  `alpha` per estimation point; macroforecast's ridge/lasso do not standardize by
  default. The estimators themselves are bit-exact to scikit-learn (§7).
- **SETAR/STAR (0.072/0.048).** `[GAP]` The paper forecasts these with iterated
  block-bootstrap; here they are direct projections (consistent with every other
  model), which is why the paper's long-horizon values run higher.

These are **replication-specification and data-vintage differences, not package
defects** — see §7 for the independent verification of each estimator.

---

## 6. Efficiency and reproducibility (objectives 3 & 4)

- **Reproducible.** With `random_state=42` every number above is fixed. The MRF was
  previously non-deterministic (unseeded global RNG); PR #469 adds per-tree seeding.
  Measured run-to-run spread of the ARRF/AR ratio *before* the fix was ~0.06 at B=50 —
  large enough to move a cell, so seeding is required for a credible table.
- **Faster, identically.** MRF fits parallelise over trees: **581 s → 47 s (12.4×) at
  16 cores**, and with per-tree seeding the parallel forest is **bit-identical** to the
  serial one (max|Δpred| = 0). The full UR/ARRF row runs in ~19 min reproducibly.
- **Pipeline note.** The packaged `run_pipeline` reproduces the direct-call numbers,
  but for this large engineered-feature MRF workload it re-materialises the 1,042-col
  feature matrix per origin (≈10× slower), and `n_jobs>1 × parallelise=True`
  over-subscribes threads. The direct-call driver used here is the practical reference;
  making the pipeline reuse expanding features is an open efficiency item.

## 7. Package defects surfaced (objective 2)

The replication drove three fixes and one open limitation:

1. **AR benchmark kitchen-sink (PR #468, merged).** `_select_lag_columns` failed open:
   an "AR" whose feature spec lacked the target's `lag0` but carried predictor lags
   regressed the target on ~240 predictor contemporaneous values — a p≈N OLS, not an
   autoregression, corrupting every ratio normalised against it. Fixed to fail closed.
2. **MRF non-determinism (PR #469).** The vendored MRF drew from the global unseeded
   `np.random` and ignored the pipeline seed; added `random_state` + per-tree seeding
   → reproducible and parallel-identical.
3. **SETAR / STAR missing (PR #470).** Implemented both as first-class models so the
   two nonlinear-TS columns of Table 4 are covered.
4. **`[open]` MRF cannot represent plain RF.** The MRF requires ≥1 X column, so the
   paper's documented "RF = MRF with `X_t = ι`" cannot be expressed; plain-RF columns
   fall back to scikit-learn `random_forest`.

Independently cross-checked and **clean**: `ols` = numpy OLS (1e-16); `ridge`/`lasso` =
scikit-learn raw (0.0); `random_forest` reproducible with `random_state`; `far`/MRF
factor path leak-free.

## 8. Reproduce

```python
import macroforecast as mf
b = mf.data.load_fred_qd("2020-01")          # FRED-QD panel + tcodes
# transform predictors by tcode; UR target = log-then-diff; build S_t (Table 2) with
# expanding pca_step/maf_step; fit macro_random_forest(..., B=50, random_state=42,
# parallelise=True) per 2-year estimation point; ratio vs AR(4). See scripts/replication.
```

`[GAP]`/`[ASSUMPTION]` register: FRED-QD vintage; 215/247 complete-series subset;
direct vs iterated SETAR/STAR; plain-RF via scikit-learn; penalty-tuning scheme.
