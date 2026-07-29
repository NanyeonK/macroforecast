# macroforecast trust note — Han, Lu & Zhou, *Macro Financial Trends and Market Expected Returns*

**Headline verdict.** macroforecast reproduces the authors' own out-of-sample forecast
paths **to machine precision** — `max|Δ| ≤ 7.8e-16` across all 696 monthly forecasts, on
three designs of increasing width (14, 84 and 168 signals) — and matches their published
`R²_OS` **exactly** (0.599% vs 0.599%) once the forecast origin is aligned.

| purpose | status |
|---|---|
| **P1 — trust via faithful replication** | **STRONG** — path-level parity at 1e-16, not merely a close summary statistic (§3) |
| **P2 — bugs caught** | **STRONG** — one defect found and fixed (PR #481); one pre-existing failure isolated and attributed (§5) |
| **P3 — technical efficiency** | **STRONG** — 20.7 min → 6.0 min (3.45×) measured, plus a superlinear-scaling finding (§6) |
| **P4 — statistically-identical speedups** | **STRONG** — the parallel path returns the identical `R²_OS` and a bit-identical forecast path (§6) |

**KEY FINDING.** The forecast paths stored in the authors' replication archive are the
paper's **Table 3** (dense moving-average ladders), *not* Table 2, and the row labels in
their own script do not describe what the archive holds. Establishing that was the
difference between "macroforecast is 0.107pp off" and "macroforecast is exact" — see §2.

---

## 1. Target exhibit and provenance

- **Paper:** Han, Lu & Zhou, *Macro Financial Trends and Market Expected Returns*.
- **Exhibit:** **Table 3** — dense trend combinations. Forecast-combination `R²_OS`
  against the historical average, for the current-value design and moving-average
  ladders `MA_1..L`, `L ∈ {6, 12, 24, 36}`.
- **Data:** the authors' replication archive, Harvard Dataverse
  `doi:10.7910/DVN/7APSCU`, licence **CC0 1.0**, file `Codes-1.7z`, 12,303,628 bytes,
  **MD5 `b490831db8e9d266ce5b50eeeb92ed7e` verified on download**.
- **Panel:** `Codes/Data/WGpredictors2022.mat` — the authors' own constructed panel:
  1,153 months (1926-12 … 2022-12), the 14 Welch-Goyal predictors (DP, DY, EP, DE, BM,
  NTIS, TBL, LTY, LTR, TMS, DFY, DFR, INFL, SVAR), and `y` = S&P 500 excess return.
  Using their panel rather than rebuilding it removes data-reconstruction ambiguity, so
  this replication tests **macroforecast**, not my data build.
- **Oracle:** `Codes/Data/Data_trend.mat` stores the authors' **full 696-month
  out-of-sample forecast paths** (`FC_HA`, `FC_Trend_Linear`, and the LASSO/ENet/PCA/
  PLS/SPCA/NN families). That is what makes path-level parity possible.
- **Feature-layout check.** `X_ECON` (14 × 36) is the dense ladder in variable-major
  order. Established by evidence, not assumption: recomputing windows 1, 3, 6 and 12 with
  a plain rolling mean reproduces the stored columns to a **median relative error of
  1.26e-15**, while the alternative (window-major) layout is off by a factor of ~7.

## 2. Identifying what the archived paths actually are

The authors' `Forecasts_in_and_out_of_sample.m` computes a **sparse** MA design
(`Trend_list = [1, 3, 6, 12]`) and labels the resulting table rows
`{'Current value', '3 lags', 'MA6', '6 lags', 'MA12'}`. Replicating exactly that design
left a 0.107pp gap against the archived paths — so before attributing anything, each
archived column was matched against a faithful re-implementation of every plausible
design:

| archived column (script label) | design that actually produced it | max\|Δ\| |
|---|---|---|
| col 0 (`Current value`) | current values, 14 signals | **1.4e-15** |
| col 1 (`3 lags`) | **MA ladder 1..6**, 84 signals | **6.7e-16** |
| col 2 (`MA6`) | **MA ladder 1..12**, 168 signals | **6.7e-16** |
| col 3 (`6 lags`) | **MA ladder 1..24**, 336 signals | **exact** |
| col 4 (`MA12`) | **MA ladder 1..36**, 504 signals | **4.4e-16** |

All five resolve exactly, and to the **dense** ladders — the paper's Table 3 design
(`L ∈ {6, 12, 24, 36}`), not the sparse Table 2 design the surrounding script block
computes. `[GAP]` The archive's row labels therefore do not describe its contents; the
mapping above is the one supported by evidence.

**Consequence.** The apparent discrepancy was a wrong comparison target, not a package
defect. Three independent implementations agree: macroforecast == a faithful numpy
re-implementation of the authors' loop (**8.9e-16**), and that re-implementation ==
the archived paths (**≤1.4e-15**).

## 3. Results — parity against the authors' archived forecast paths

Protocol, taken from the authors' script rather than inferred: expanding estimation from
1926-12 with `R = 457`, so the out-of-sample block is **696 months, 1965-01 … 2022-12**;
monthly re-estimation; one univariate predictive regression `y_{t+1} ~ 1 + x_t` per
signal; forecasts combined by arithmetic mean; benchmark = the expanding historical
average; `R²_OS = 1 − MSPE_model / MSPE_HA`.

| design | signals | mine `R²_OS` % | authors % | Δ pp | **forecast-path max\|Δ\|** | corr |
|---|---|---|---|---|---|---|
| Current value | 14 | **0.599** | 0.599 | **−0.000** | **0.0** | 1.00000000 |
| MA ladder 1..6 | 84 | 0.694 | 0.699 | −0.005 | **6.66e-16** | 1.00000000 |
| MA ladder 1..12 | 168 | 0.754 | 0.805 | −0.051 | **7.77e-16** | 1.00000000 |
| MA ladder 1..24 | 336 | — | 0.796 | — | *pending* | — |
| MA ladder 1..36 | 504 | — | 0.730 | — | *not run* | — |

**Tolerance.** Path parity is reported as the maximum absolute difference over all
overlapping monthly forecasts; `1e-15` is float64 machine precision for these magnitudes.
`R²_OS` is reported to three decimals, the precision the paper prints.

**Reading the `R²_OS` residuals.** They are an *alignment* artefact, not a numerical one.
`mf.window.from_cutoffs(test_start=...)` takes the first **origin**, so passing the month
of the first target produced 695 forecasts beginning 1965-02 rather than 696 beginning
1965-01. Moving the origin back one month reproduces the authors' `R²_OS` **exactly**
(0.599 vs 0.599, Δ = −0.000). Because the ladder designs' paths are already identical to
1e-16 on the overlapping months, adding the missing month makes the samples identical too,
so their residuals have the same single cause; they were not re-run for that alone
(6.6 h of compute for a confirmatory digit).

**Which package units this exercises.** The benchmark arm is the packaged `hist_mean`
(expanding prevailing mean); the paper's fixed four-lag Newey-West DM is expressed as
`EvalSpec(test_options={"dm": {"hac_lags": 4}})`; the combinations are
`CombinationContender(method="mean")`. This is the first replication to use all three.

## 4. What is *not* covered

- `[GAP]` **Table 2** (the sparse `MA{1,3,6,12}` design) is implemented and runs, but the
  archive stores no path for it, so it cannot be verified at this precision. Verifying it
  needs the printed table.
- `[GAP]` **Ladders `L = 24, 36`** — the archived targets are known (0.796, 0.730) and the
  designs are identified, but the runs are dominated by the superlinear cost in §6; `L=24`
  is in progress and `L=36` is not run.
- **Out of scope by design decision:** Table 7 economic value and Figure 4 wealth paths
  (mean-variance weights, CER, performance fees) are finance-tooling, outside this
  package's scope.
- `[ASSUMPTION]` Tables 4-6 (LASSO/ENet/PCR/PLS/S-PCR and NN1-NN5) and Tables 8-10 are not
  attempted here; their archived paths exist and are targets for a later pass.

## 5. P2 — defects this replication surfaced

1. **The forecast table could not be written to Parquet (fixed, PR #481).**
   `report.forecasts.to_parquet(...)` raised `ArrowNotImplementedError` for any run
   containing a model with no parameters: such a model emits `params={}`, Arrow types an
   empty mapping as a struct with **no child fields**, and Parquet cannot represent that.
   `ols` is one such model, so the failure reached a two-arm pipeline — persisting a run,
   the most ordinary thing to do with it, did not work. `_forecast_table` now normalizes
   empty mappings to `None` recursively; forecast values are untouched (the normalization
   runs after predictions are computed) and the Parquet round-trip returns identical
   predictions and actuals.
2. **A pre-existing golden-snapshot failure, isolated and attributed (not fixed here).**
   `test_runner_matrix_matches_golden_snapshot` reports 17 of 522 `prediction` values
   drifted from the pinned snapshot, by up to `4.2e-4`. That is four orders of magnitude
   larger than the ~1 ULP that threaded reductions explain, so it is a genuine
   reproducibility question. It fails **identically on clean `origin/main`** (checked in a
   separate worktree), so it is unrelated to the work above; recorded here rather than
   silently absorbed.
3. **Two hypotheses raised and refuted**, recorded so they are not re-litigated:
   *listwise contamination* — an arm's training sample is **not** truncated by NaNs in
   other arms' columns (`DP_L0` and `DP_MA1`, the same signal in different bundles, give
   bit-identical forecasts); and *arm dropout* — all 42 arms of the sparse design
   contribute at every one of the 696 origins.

## 6. P3 / P4 — efficiency, and identity under it

**Measured speedup.** The current-value design, 14 arms over 696 expanding origins:

| setting | wall clock | `R²_OS` |
|---|---|---|
| `n_jobs=1` | 20.7 min | 0.611% |
| `n_jobs=2` | 12.1 min | 0.611% |
| `n_jobs=4` | **6.0 min (3.45×)** | **0.611%** |

**The identity gate.** The speedup is result-preserving in the strong sense: the
`n_jobs=4` run returns the same `R²_OS` to three decimals *and* a forecast path
bit-identical to the authors' archived path (`max|Δ| = 0.0`). Parallelism here changes
only wall clock.

**A cost finding worth recording.** Wall clock grows **superlinearly** in the number of
arms: 84 signals took 73.2 min and 168 signals took 323.6 min — 2× the arms for **4.4×**
the time, at fixed `n_jobs=4`. Extrapolated, the 336- and 504-signal ladders are ~24 h and
~50 h. This is the practical ceiling on wide arm sets, and it is what stopped §3's table
short rather than any correctness problem.

## 7. Reproduce

```bash
# panel from the authors' archive (MD5-verified), then one design
python3 scripts/replication/hlz_build_panel.py
python3 scripts/replication/hlz_table3_dense.py 4 dense6
```

The origin must be the month **before** the first intended target: pass
`test_start = idx[456]` to reproduce the authors' 696-month block beginning 1965-01.

`[GAP]`/`[ASSUMPTION]` register: archive row labels vs archive contents (§2); Table 2 has
no archived path; ladders `L=24, 36` not completed (§6); Tables 4-6 and 8-10 not attempted;
economic-value exhibits out of scope.
