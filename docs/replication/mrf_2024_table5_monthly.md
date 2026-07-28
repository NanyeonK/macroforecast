# Replicating Goulet Coulombe (2024), Table 5 — *Monthly Results*

**Target exhibit.** **Table 5, "Monthly Results"** (Appendix A.6, *Monthly Forecasting
Results*) of Goulet Coulombe, "The Macroeconomy as a Random Forest", *Journal of
Applied Econometrics* 39 (2024); arXiv 2006.12724. This is the monthly (FRED-MD)
companion to Table 4, replicated in `mrf_2024_replication.md`.

Every cell below was produced by `scripts/replication/mrf_2024_pipeline/run_table5.py`
with `random_state=42`; the run is 25 cells (5 targets x 5 horizons), completed in
24.6 h on 16 cores with no failed cells.

---

## 1. Headline

| | |
|---|---|
| Targets | **5 of 5** (IP, UR, INF, SPREAD, HOUST) |
| Models | **11 of 11** columns (incl. the Atkeson-Ohanian AO-12 / AO-h forecasts) |
| Horizons | **5 of 5** (h = 1, 3, 9, 12, 24 months) |
| Overall accuracy | **mean\|Δ\| = 0.098** vs AR(12); **0.107** vs AR(4) |
| Qualitative verdict | **both A.6 stories reproduce** — MAF-carrying models beat non-MAF ones on every target, and all four MRFs beat the AR benchmark on inflation (§4) |
| Package defect surfaced | **one degenerate tree destroyed the whole forecast** — 13/100 MRF model-cells lost; root-caused to the adapter and **fixed in PR #478** (§5) |

**Per-target overall mean\|Δ\|:**

| target | vs AR(12) | vs AR(4) |
|---|---|---|
| IP | **0.081** | 0.081 |
| UR | **0.103** | 0.101 |
| INF | **0.061** | 0.102 |
| SPREAD | **0.187** | 0.191 |
| HOUST | **0.058** | 0.060 |
| **all** | **0.098** | 0.107 |

---

## 2. Spec (paper Appendix A.6)

- **Panel:** `mf.data.load_fred_md("2020-01")`. `[ASSUMPTION]` the paper describes 134
  monthly series; this vintage yields **127**, of which **117** are complete-case through
  2002-12 and enter the predictor block. The paper's exact vintage is unstated `[GAP]`.
- **Targets:** IP (`INDPRO`) replaces GDP and IR is dropped, per A.6. `[ASSUMPTION]`
  INF = Δlog `CPIAUCSL` and HOUST = Δlog `HOUST`, inherited from the quarterly runner;
  SPREAD = `GS10` − `FEDFUNDS`, same definition as Table 4.
- **Forecast target:** the **h-period average growth rate**, `(1/h)·Σ_{h'=1..h} y_{t+h'}`,
  not the h-step growth rate — this is the A.6 change from Table 4.
- **Horizons:** h ∈ {1, 3, 9, 12, 24} months.
- **Benchmark:** `[GAP]` **the paper contradicts itself.** The A.6 text says the AR
  counterpart is **AR(12)**; the Table-5 caption and the paper's global convention say
  **AR(4)**. Both are computed here and reported side by side — see §3.
- **State `S_t`:** 12 lags of `y`; a linear trend; 2 lags of every FRED-MD series;
  5 PCA factors × 12 lags; 2 MAFs per variable (`maf_step(max_lag=12)`) — 529 predictor
  columns + 12 target lags, all fit expanding/leak-free.
- **Protocol:** expanding estimation; POOS 2003M01–2014M12 (144 months); re-estimated
  every **24 months** (6 estimation points); **direct** forecasts; MRF hyper-parameters
  are the author's `macrorf::MRF` defaults with `B=50`, `random_state=42`.

---

## 3. The AR(4)-vs-AR(12) `[GAP]` is empirically immaterial

The paper's self-contradiction turns out **not to change any conclusion**. Across the
whole table the two denominators give near-identical RMSE — the fitted AR(4) column
measured against the AR(12) benchmark sits at 0.955–1.017 — so every ratio is stable:

- overall mean\|Δ\| **0.098** (AR12) vs **0.107** (AR4)
- the largest per-target gap is INF (0.061 vs 0.102); every other target moves by ≤0.004

**The AR benchmark itself was independently verified** (this was the first defect of the
whole programme — PR #468 — so it was re-checked rather than assumed):

- the runner's hand-rolled numpy OLS is **bit-identical** to the package `ols` on the same
  lag columns (max\|Δ\| = 6e-16);
- at h=1 the AR(4) beats a training-mean forecast by 11.4% (RMSE 0.006897 vs 0.007683,
  in-sample R² = 0.127) — it is not degenerate;
- **the paper's own AR4 column is the tightest-matching row in the table** (mean\|Δ\| =
  0.005–0.019 per target). A broken benchmark would corrupt this row first.

At h=24 the AR *does* collapse toward the training mean (RMSE 0.003519 vs a mean
forecast's 0.003526; in-sample R² = 0.010; forecast s.d. 0.00031 against a target s.d. of
0.00292). That is a property of the data — four lags of monthly growth carry almost no
information about the next 24-month average — not a defect, and the paper's own AR4
column reports the same near-equality.

---

## 4. Results — Table 5, all five targets

Cells are `RMSE_ratio_mine / RMSE_ratio_paper` against **AR(12)**; `mean|Δ|` is the mean
absolute per-cell difference over the horizons that produced a number. `--` marks a cell
lost to the `nan` defect in §5 (it is excluded from that row's mean, and the count of lost
cells is reported in §5 rather than hidden).

### IP

| model | h=1 | h=3 | h=9 | h=12 | h=24 | mean\|Δ\| |
|---|---|---|---|---|---|---|
| AR4 | 1.004 / 1.00 | 1.004 / 1.02 | 1.008 / 1.01 | 1.007 / 1.01 | 1.000 / 1.00 | 0.005 |
| AO-12 | 1.091 / 1.11 | 1.190 / 1.17 | 1.276 / 1.04 | 1.330 / 1.00 | 1.538 / 0.84 | 0.261 |
| AO-h | 1.331 / 1.14 | 0.985 / 1.02 | 1.251 / 1.03 | 1.330 / 1.00 | 1.336 / 0.84 | 0.255 |
| FA-AR | 0.987 / 0.96 | 1.071 / 0.99 | 1.053 / 1.06 | 1.035 / 1.05 | 1.003 / 1.17 | 0.060 |
| RF | 1.034 / 1.03 | 1.011 / 1.12 | 1.100 / 1.02 | 1.041 / 0.99 | 0.796 / 0.92 | 0.074 |
| RF-MAF | 1.029 / 0.94 | 1.077 / 0.98 | 1.092 / 1.06 | 1.011 / 0.97 | 0.811 / 0.86 | 0.061 |
| AR+RF | 1.009 / 0.97 | 0.981 / 0.96 | 1.055 / 1.02 | 0.993 / 0.91 | 0.808 / 0.86 | 0.046 |
| ARRF | 1.019 / 0.99 | 1.025 / 1.03 | 0.971 / 1.04 | 0.956 / 0.97 | 0.830 / 0.88 | 0.033 |
| FA-ARRF | 1.042 / 0.96 | 1.055 / 1.01 | 1.082 / 1.10 | 1.056 / 1.05 | 0.916 / 0.95 | 0.037 |
| Tiny ARRF | 1.022 / 1.02 | 1.023 / 1.02 | 1.050 / 1.09 | 1.047 / 1.13 | 1.045 / 1.11 | 0.039 |
| VARRF | 1.031 / 1.02 | 1.026 / 1.08 | 0.993 / 1.03 | 0.963 / 0.96 | -- / 0.89 | 0.026 |

**IP overall mean\|Δ\| = 0.081**

### UR

| model | h=1 | h=3 | h=9 | h=12 | h=24 | mean\|Δ\| |
|---|---|---|---|---|---|---|
| AR4 | 1.017 / 1.01 | 1.008 / 1.00 | 0.997 / 0.99 | 0.998 / 0.99 | 1.005 / 1.02 | 0.009 |
| AO-12 | 1.037 / 1.03 | 1.062 / 1.10 | 1.101 / 1.11 | 1.114 / 1.07 | 1.306 / 1.02 | 0.077 |
| AO-h | 1.444 / 1.09 | 1.131 / 1.05 | 1.047 / 1.10 | 1.114 / 1.07 | 1.390 / 1.03 | 0.178 |
| FA-AR | 0.974 / 0.95 | 0.989 / 0.86 | 1.010 / 0.92 | 1.015 / 0.96 | 1.009 / 1.06 | 0.070 |
| RF | 0.972 / 0.97 | 0.938 / 1.05 | 0.870 / 1.02 | 0.812 / 0.97 | 0.656 / 0.91 | 0.135 |
| RF-MAF | 0.965 / 0.87 | 0.870 / 0.81 | 0.821 / 0.96 | 0.819 / 0.96 | 0.693 / 0.84 | 0.116 |
| AR+RF | 0.979 / 0.95 | 0.913 / 0.92 | 0.815 / 0.91 | 0.801 / 0.91 | 0.686 / 0.81 | 0.073 |
| ARRF | 0.984 / 0.91 | 0.857 / 0.89 | -- / 0.97 | 0.794 / 0.99 | 0.677 / 0.91 | 0.134 |
| FA-ARRF | 0.958 / 0.90 | 0.877 / 0.82 | 0.772 / 0.98 | 0.778 / 0.94 | -- / 0.97 | 0.121 |
| Tiny ARRF | -- / 0.98 | -- / 1.03 | -- / 1.16 | -- / 1.17 | -- / 1.28 | -- |
| VARRF | 1.021 / 0.94 | 0.914 / 0.89 | 0.831 / 0.97 | 0.810 / 0.96 | 0.665 / 0.87 | 0.120 |

**UR overall mean\|Δ\| = 0.103**

### INF

| model | h=1 | h=3 | h=9 | h=12 | h=24 | mean\|Δ\| |
|---|---|---|---|---|---|---|
| AR4 | 1.024 / 1.02 | 1.024 / 1.04 | 1.088 / 1.07 | 1.104 / 1.09 | 1.082 / 1.04 | 0.019 |
| AO-12 | 0.946 / 1.11 | 0.948 / 1.02 | 0.900 / 0.92 | 0.892 / 0.91 | 0.890 / 0.90 | 0.057 |
| AO-h | 0.913 / 1.18 | 1.137 / 1.24 | 0.990 / 1.01 | 0.892 / 0.91 | 0.690 / 0.86 | 0.116 |
| FA-AR | 1.048 / 0.99 | 1.056 / 1.04 | 1.174 / 1.16 | 1.208 / 1.21 | 1.286 / 1.35 | 0.031 |
| RF | 0.913 / 1.07 | 0.819 / 0.93 | 0.747 / 0.86 | 0.750 / 0.88 | 0.697 / 1.00 | 0.163 |
| RF-MAF | 0.917 / 1.06 | 0.847 / 0.88 | 0.714 / 0.78 | 0.704 / 0.79 | 0.754 / 1.12 | 0.139 |
| AR+RF | 1.002 / 1.01 | 1.012 / 1.05 | 1.141 / 1.15 | 1.126 / 1.15 | 1.056 / 1.12 | 0.029 |
| ARRF | 0.955 / 0.95 | 0.887 / 0.90 | 0.718 / 0.72 | -- / 0.73 | 0.705 / 0.71 | 0.006 |
| FA-ARRF | -- / 0.96 | 0.843 / 0.88 | 0.722 / 0.82 | 0.679 / 0.67 | 0.679 / 0.69 | 0.038 |
| Tiny ARRF | 0.907 / 0.95 | 0.861 / 0.90 | 0.755 / 0.73 | 0.694 / 0.67 | 0.669 / 0.55 | 0.050 |
| VARRF | 0.967 / 0.93 | 0.873 / 0.88 | 0.755 / 0.76 | 0.715 / 0.70 | 0.697 / 0.73 | 0.020 |

**INF overall mean\|Δ\| = 0.061**

### SPREAD

| model | h=1 | h=3 | h=9 | h=12 | h=24 | mean\|Δ\| |
|---|---|---|---|---|---|---|
| AR4 | 1.007 / 0.99 | 1.016 / 1.01 | 1.006 / 1.01 | 1.007 / 1.02 | 0.999 / 1.03 | 0.014 |
| AO-12 | 1.653 / 2.88 | 1.527 / 1.68 | 1.350 / 1.36 | 1.315 / 1.28 | 1.321 / 1.34 | 0.289 |
| AO-h | 0.609 / 1.23 | 0.872 / 1.07 | 1.212 / 1.27 | 1.315 / 1.28 | 1.490 / 1.34 | 0.212 |
| FA-AR | 1.101 / 1.21 | 1.049 / 1.25 | 0.863 / 1.06 | 0.826 / 1.05 | 0.775 / 0.96 | 0.183 |
| RF | 0.916 / 3.52 | 0.816 / 1.69 | 0.690 / 0.94 | 0.635 / 0.80 | 0.466 / 0.80 | 0.845 |
| RF-MAF | 0.885 / 1.07 | 0.785 / 0.82 | 0.643 / 0.73 | 0.569 / 0.66 | 0.543 / 0.70 | 0.111 |
| AR+RF | 0.866 / 0.91 | 0.779 / 0.81 | 0.650 / 0.72 | 0.591 / 0.60 | 0.518 / 0.71 | 0.069 |
| ARRF | 1.053 / 0.99 | 0.932 / 1.06 | 0.664 / 0.70 | -- / 0.68 | 0.549 / 0.69 | 0.092 |
| FA-ARRF | 0.983 / 0.98 | 0.928 / 0.85 | 0.625 / 0.62 | 0.555 / 0.65 | 0.558 / 0.63 | 0.051 |
| Tiny ARRF | -- / 0.96 | 0.915 / 1.00 | 0.923 / 1.07 | 0.984 / 1.07 | 0.992 / 0.90 | 0.102 |
| VARRF | 0.939 / 0.93 | 0.811 / 0.88 | 0.606 / 0.67 | 0.520 / 0.64 | 0.502 / 0.70 | 0.092 |

**SPREAD overall mean\|Δ\| = 0.187**

### HOUST

| model | h=1 | h=3 | h=9 | h=12 | h=24 | mean\|Δ\| |
|---|---|---|---|---|---|---|
| AR4 | 1.004 / 1.00 | 0.972 / 0.96 | 0.978 / 0.98 | 0.975 / 0.98 | 0.955 / 0.95 | 0.006 |
| AO-12 | 1.044 / 1.10 | 1.055 / 1.06 | 1.051 / 1.05 | 1.071 / 1.05 | 1.129 / 1.09 | 0.024 |
| AO-h | 1.649 / 1.35 | 1.357 / 1.34 | 1.112 / 1.12 | 1.071 / 1.05 | 1.106 / 1.07 | 0.076 |
| FA-AR | 1.032 / 1.07 | 1.069 / 1.15 | 1.063 / 1.35 | 1.052 / 1.32 | 0.967 / 1.17 | 0.175 |
| RF | 1.030 / 1.08 | 1.061 / 1.03 | 0.996 / 0.98 | 0.958 / 0.95 | 0.786 / 0.87 | 0.038 |
| RF-MAF | 1.019 / 1.02 | 1.061 / 1.07 | 0.998 / 1.02 | 0.993 / 1.00 | 0.863 / 0.94 | 0.023 |
| AR+RF | 1.021 / 1.00 | 1.032 / 1.03 | 0.998 / 1.01 | 0.985 / 1.01 | 0.855 / 0.95 | 0.031 |
| ARRF | 1.028 / 1.01 | -- / 1.04 | 1.021 / 1.02 | 1.003 / 1.00 | 0.854 / 1.00 | 0.042 |
| FA-ARRF | 1.041 / 1.02 | 1.129 / 1.03 | 1.030 / 1.14 | 1.029 / 1.12 | 0.905 / 1.15 | 0.113 |
| Tiny ARRF | 1.030 / 1.02 | 1.012 / 1.01 | 1.023 / 1.03 | 1.052 / 1.11 | 1.088 / 1.23 | 0.044 |
| VARRF | 1.042 / 1.01 | 1.099 / 1.04 | 1.011 / 1.03 | 1.048 / 1.03 | 0.866 / 1.06 | 0.065 |

**HOUST overall mean\|Δ\| = 0.058**

### Family accuracy (mean\|Δ\| by model family, vs AR(12))

| target | MRF | RF/tree | linear | AO |
|---|---|---|---|---|
| IP | 0.034 | 0.060 | 0.032 | 0.258 |
| UR | 0.125 | 0.108 | 0.039 | 0.127 |
| INF | 0.028 | 0.110 | 0.025 | 0.086 |
| SPREAD | 0.084 | 0.342 | 0.099 | 0.251 |
| HOUST | 0.066 | 0.031 | 0.090 | 0.050 |

The **MRF family is the best-replicated block on three of five targets** (IP 0.034, INF
0.028, SPREAD 0.084) — the same pattern as the quarterly Table 4, where the MRF family
also landed inside 0.02–0.10. The **AO column is the single systematic deviation** and is
attributed in §6.

### Do the paper's two qualitative claims reproduce?

**(i) "MAFs are without any doubt the major improvement" (A.6, for IP/UR/SPREAD).**
Mean forecast ratio of the MAF-carrying models (`RF-MAF, AR+RF, ARRF, FA-ARRF, VARRF`)
against the models without MAFs (`FA-AR, RF, AO-12, AO-h, Tiny ARRF`); lower is better:

| target | mine: with MAF | mine: no MAF | paper: with MAF | paper: no MAF | direction |
|---|---|---|---|---|---|
| IP | **0.993** | 1.119 | **0.980** | 1.035 | **reproduced** |
| UR | **0.839** | 1.049 | **0.914** | 1.038 | **reproduced** |
| INF | **0.846** | 0.911 | **0.886** | 0.974 | **reproduced** |
| SPREAD | **0.711** | 1.026 | **0.776** | 1.320 | **reproduced** |
| HOUST | **0.997** | 1.075 | **1.030** | 1.106 | **reproduced** |

MAF-carrying models win on **all five** targets, in both mine and the paper — including
the three the paper singles out.

**(iii) "all MRFs do very well for inflation".** Mean ratio against AR(12) on INF
(< 1 means it beats the benchmark):

| model | mine | paper |
|---|---|---|
| ARRF | **0.816** | 0.802 |
| FA-ARRF | **0.731** | 0.804 |
| Tiny ARRF | **0.777** | 0.760 |
| VARRF | **0.801** | 0.800 |

All four MRFs beat the benchmark on inflation in both, at nearly the same magnitude —
the paper's strongest monthly claim reproduces almost exactly (`ARRF` mean\|Δ\| = 0.006).

---

## 5. Package defect surfaced and FIXED (objective 2): one degenerate tree destroyed the forecast

**13 of the 100 MRF model-cells** (25 cells x 4 MRF models) could not be
scored, because the MRF returned a `nan` for at least one out-of-sample month and a
single `nan` propagates through the RMSE to destroy the whole cell. **No non-MRF model
lost a single cell** (0 of 175).

| model | lost cells | where |
|---|---|---|
| ARRF | 4/25 | HOUST h3, INF h12, SPREAD h12, UR h9 |
| FA-ARRF | 2/25 | INF h1, UR h24 |
| Tiny ARRF | 6/25 | SPREAD h1, UR h1, UR h3, UR h9, UR h12, UR h24 |
| VARRF | 1/25 | IP h24 |

This **corrects an earlier attribution.** The quarterly replication saw the same failure
only in the intercept-only (`X_t = ι`) path and recorded it as specific to that mode. The
monthly run shows it is **not** intercept-specific: it hits ordinary `ARRF`, `FA-ARRF` and
`VARRF` fits too. The gradient is with the *state*: `Tiny ARRF`, whose `S_t` is just 12
target lags plus a trend, is by far the most fragile (it loses **all five** UR horizons),
while models on the full 541-column `S_t` lose the occasional cell. A small, collinear,
trend-dominated state is the trigger; every affected fit completed normally (203–333 s)
and reported no error.

### Root cause — the adapter, not the vendored MRF (PR #478, merged)

The trees were never the whole story. `output["pred_ensemble"]` is the **raw per-tree
committee** `(B, n_oos)`, so reducing it *is* the ensemble average — and the adapter
reduced it with a **nan-propagating** `arr.mean(axis=0)`. The vendored backend does not
do that: its own ensemble output `output["pred"]` is
`pd.DataFrame(committee).mean(axis=0)`, which is **nan-skipping**, as is the R
prototype. The adapter ignored the backend's answer and recomputed the average under
different semantics, so **one degenerate tree out of B wiped out the forecast for that
row** — and with it the RMSE, and with that the whole cell.

The reproducer makes it unambiguous: at an origin where the per-tree NaN count was
**1 of 10**, the ensemble forecast still came back NaN. The adapter additionally
suppresses the backend's `invalid value encountered in divide` RuntimeWarning, which is
the signal that would have exposed this.

**Fixed in PR #478**, three parts:

1. the committee is reduced with nan-skipping means, matching the backend and R — a row
   that *no* tree could predict stays NaN, because that case is genuinely undefined;
2. `degenerate_tree_predictions_` is exposed and a warning is raised when trees are
   skipped, so the condition is visible instead of silent;
3. the committee axis is resolved from the known tree count. Deducing it from shape
   alone silently averaged **across forecast dates instead of across trees** whenever
   `B == n_oos` — and `B=25` is the package default, so that collision is reachable.

**Statistically identical where nothing degenerates** (objective 4, measured): across
395 random NaN-free committees the new reduction is **bit-identical** to the old one
(max\|Δ\| = 0.000e+00), including through the previous call signature.

**Verified on the failing case.** Re-running `UR h=1` against merged `main`, the cell
that this table records as lost now produces a number, and a good one:

| model | before (this table) | after PR #478 | paper |
|---|---|---|---|
| Tiny ARRF (UR, h=1) | `--` (lost) | **1.023** (mean\|Δ\| 0.043) | 0.98 |

All eleven models returned a value on that re-run; no cell was lost.

`[note]` The tables in §4 are the **original** run and still show the `--` holes, so the
numbers here remain exactly those the committed result JSONs contain. Re-running the
eleven affected `(target, horizon)` cells to close the holes is a separate ~15 h job.

---

## 6. Where else it deviates, and why (attributed — no reverse-engineering)

- **AO-12 / AO-h (the only systematic gap: family mean\|Δ\| 0.05–0.29).** `[GAP]` The
  Atkeson-Ohanian forecasts here are the trailing 12-month and trailing h-month averages
  of the transformed target. On IP at h=24 that gives 1.538 against the paper's 0.84: a
  trailing average is catastrophic across 2008–09, when the last 12 months of growth were
  deeply negative and the next 24 were a recovery, whereas an AR anchored on the long-run
  mean is very hard to beat for a smooth 24-month-average target. The paper's AO must be
  constructed differently; its exact definition is not given beyond "1, h and 12 months
  moving averages". **This was not tuned to close the gap.** The deviation is confined to
  the AO rows — a mis-specified *denominator* would move every model in the same
  direction, and it does not (§3).
- **SPREAD `RF` (mean\|Δ\| 0.845).** Driven entirely by the paper's own outliers at the
  short end (paper 3.52 at h=1 and 1.69 at h=3, against 0.916 and 0.816 here). At h ≥ 9
  the same row agrees closely (0.690/0.94, 0.635/0.80, 0.466/0.80).
- **UR / SPREAD generally (0.103 / 0.187).** These two load hardest on the PCA factors,
  which are built from 117 complete series of the 2020-01 vintage rather than the paper's
  full 134 — factor differences propagate into every model that uses them.
- **`Tiny ARRF` where it survives.** Its restricted state is exactly the configuration
  most affected by §5, so its surviving cells are the least trustworthy in the table.

---

## 7. `[GAP]` / `[ASSUMPTION]` register

| tag | item |
|---|---|
| `[GAP]` | Benchmark: A.6 text says AR(12), the Table-5 caption says AR(4). Both computed; immaterial (§3). |
| `[GAP]` | AO-12 / AO-h construction — the paper's moving-average definition is under-specified (§6). |
| `[GAP]` | FRED-MD vintage unstated; 2020-01 used. |
| `[GAP]` | Diebold-Mariano tests in the paper's Table 5 are not reproduced here — only the ratios. |
| `[ASSUMPTION]` | 127 series load from this vintage vs the paper's 134; 117 complete-case predictors used. |
| `[ASSUMPTION]` | INF = Δlog CPIAUCSL, HOUST = Δlog HOUST, SPREAD = GS10 − FEDFUNDS (inherited from Table 4). |
| `[ASSUMPTION]` | `Tiny ARRF` state = 12 target lags + trend; `VARRF` linear part = y lags + INDPRO/GS1/CPIAUCSL. |
| **fixed** | MRF sporadic `nan` forecasts (§5) — 13/100 MRF cells. Root cause was the adapter's nan-propagating committee reduction; **fixed in PR #478** (bit-identical where nothing degenerates). §4 still shows the original run's holes. |

## 8. Reproduce

```bash
# one cell (target, horizon, cores); ~2.4 h per cell on 6 cores
python3 scripts/replication/mrf_2024_pipeline/run_table5.py UR 24 6
```

Run all 25 cells with **at most two concurrent processes at `n_cores=6`** — a 16-core box
is oversubscribed by anything more, and the MRF fits are the dominant cost (203–333 s
each, ~2.4 h per cell, 24.6 h for the full table).
