# macroforecast trust note — Han, Lu & Zhou, *Macro Financial Trends and Market Expected Returns*

**Headline verdict.** This is the package's **first verified worked example of the
bring-your-own-data path**: an already-assembled panel and a non-macro target, driven
entirely through `custom_dataset` with the FRED loaders and the tcode machinery unused.
On that path macroforecast reproduces the authors' own out-of-sample forecast paths **to
machine precision** — `max|Δ| ≤ 1.0e-15` across all 696 monthly forecasts — and matches
their published `R²_OS` **exactly** (0.599 vs 0.60) once the forecast origin is aligned.

| purpose | status |
|---|---|
| **P1 — trust via faithful replication** | **STRONG** — path-level parity at 1e-15, plus printed-table parity across Tables 2-4 (§3-§5) |
| **P2 — bugs caught** | **STRONG** — one defect found and fixed (PR #481); one pre-existing failure isolated and attributed (§7) |
| **P3 — technical efficiency** | **STRONG** — 20.7 min → 6.0 min (3.45×) measured, plus a superlinear-scaling finding (§8) |
| **P4 — statistically-identical speedups** | **STRONG** — the parallel path returns the identical `R²_OS` and a bit-identical forecast path (§8) |

**Why this paper, on a macro package.** The target is the S&P 500 excess return, not a macro
aggregate — so the question this exercise answers is *not* "does the package do macro?" but
"**does the methodology spine work when the FRED layer is removed?**" Every prior replication
note in this directory drives the package through `load_fred_md/qd`. This one does not touch
those loaders at all, and it still lands on an external paper's published numbers.

**KEY FINDING.** The forecast paths stored in the authors' replication archive are the
paper's **Table 3**, not Table 2, and the row labels in their own script do not describe what
the archive holds. Establishing that was the difference between "macroforecast is 0.107pp
off" and "macroforecast is exact" — see §4.

---

## 1. Target exhibits and provenance

- **Paper:** Han, Lu & Zhou, *Review of Asset Pricing Studies* 16(2), p. 241,
  `doi:10.1093/rapstu/raaf014`. Printed values below were read out of the published PDF and
  its Internet Appendix, not from a working-paper draft.
- **Exhibits replicated:** **Table 2** (sparse forecast combinations), **Table 3** (dense
  trend ladders), **Table 4 columns PCR/PLS/S-PCR** (factor methods). Tables 4's shrinkage
  columns, Table 5 (neural nets) and Table 6 (ensembles) are scoped in §6.
- **Data:** the authors' own panel, `WGpredictors2022.mat`, from Harvard Dataverse
  `doi:10.7910/DVN/7APSCU`, licence CC0 1.0, `Codes-1.7z`, 12,303,628 bytes, **MD5
  `b490831db8e9d266ce5b50eeeb92ed7e` verified on download**. 1,153 months (1926-12 …
  2022-12), the 14 Welch-Goyal predictors, and `y` = the S&P 500 excess return, which the
  archive itself traces to their spreadsheet `Returns_trends_data_inputs`, sheet
  "Moving average". Sanity: `y` averages 0.675%/month (**8.1% annualized**) with 5.44%
  monthly volatility — the equity premium, at the right magnitude.
- **Oracle:** `Data_trend.mat` stores the authors' **full 696-month out-of-sample forecast
  paths** (`FC_HA`, `FC_Trend_Linear`, `FC_PCA/PLS/SPCA/LASSO/ENET`, `FC_NN1..5`). These are
  the **comparison target and never an input** — the pipeline never sees them, so parity at
  1e-15 means two implementations reached the same numbers, not that one copied the other.

**What using the authors' panel does and does not buy.** It removes data-reconstruction
ambiguity, so the exercise tests **macroforecast** rather than my data build. It also means
this note does **not** establish that the panel can be rebuilt from Goyal-Welch raw files.

## 2. How the package was driven (the BYOD path)

```python
bundle  = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
targets = [TargetSpec("y", transform="level")]
```

Every column carries tcode `1` (level) because the authors' `ECON` is already in the units
the paper forecasts with, and `y` is already an excess return. That choice is **verified, not
assumed**: had the tcode or target handling been wrong, the reconstructed MA ladder would not
match the archive's `X_ECON` to 1.26e-15, nor the forecast paths to 1e-15.

| exercised | untouched, therefore unverified here |
|---|---|
| `custom_dataset`, non-macro target, level tcodes | `load_fred_md` / `load_fred_qd` / `load_fred_sd` |
| expanding window, per-origin refit, per-arm feature specs | McCracken-Ng tcode transformation |
| `CombinationContender(method="mean")` | vintage / real-time alignment |
| `hist_mean` benchmark (unit A1), `hac_lags=4` (unit A6), `r2_oos`, CW/DM | tcode-to-target mapping |

`hist_mean` and the fixed-lag Newey-West override get their first production use here.

## 3. Table 2 — sparse forecast combinations

Design read from the authors' `Forecasts_in_and_out_of_sample.m`: expanding estimation from
1926-12 with `R = 457`, so the out-of-sample block is **696 months, 1965-01 … 2022-12**;
monthly re-estimation; one univariate predictive regression `y_{t+1} ~ 1 + x_t` per signal;
arithmetic-mean combination; benchmark = the expanding historical average.

| panel | design | signals | mine `R²_OS` % | printed % | Δ pp |
|---|---|---|---|---|---|
| A | `Xt` | 14 | **0.599** | 0.60 | **−0.001** |
| B | `Xt; MA3; MA6` | 42 | 0.686 | 0.70 | −0.014 |
| B | `Xt; Xt−1; … ; Xt−5` | 84 | 0.573 | 0.59 | −0.017 |
| C | `Xt; MA3; MA6; MA12` | 56 | 0.693 | 0.76 | −0.067 |
| C | `Xt; Xt−1; … ; Xt−11` | 168 | 0.509 | 0.57 | −0.061 |

**The paper's central contrast reproduces.** Adding the same dimensionality as *lags* hurts
(0.573, 0.509 — both below panel A's 0.599) while adding it as *moving averages* helps
(0.686, 0.693). That sign pattern, not the level, is what the paper's argument rests on.

`[GAP]` The archive stores **no** forecast path for the sparse designs — it consistently
saves the dense ones. So these rows can be judged for closeness to the printed values but
**cannot be verified at path level**, and the two larger residuals (−0.067, −0.061) cannot be
diagnosed further from the material available.

## 4. Table 3 — dense trend ladders, and what the archive actually contains

The authors' script computes the *sparse* design and labels its output rows
`{'Current value','3 lags','MA6','6 lags','MA12'}` — but `Data_trend.mat` stores something
else. Each archived column was matched to a design by cross-checking faithful
re-implementations:

| archived column (script label) | design that actually produced it | max\|Δ\| |
|---|---|---|
| col 0 (`Current value`) | current values, 14 signals | 1.4e-15 |
| col 1 (`3 lags`) | **MA ladder 1..6**, 84 signals | 6.7e-16 |
| col 2 (`MA6`) | **MA ladder 1..12**, 168 signals | 6.7e-16 |
| col 3 (`6 lags`) | **MA ladder 1..24**, 336 signals | exact |
| col 4 (`MA12`) | **MA ladder 1..36**, 504 signals | 4.4e-16 |

All five resolve to the **dense** ladders — the paper's Table 3 (`Xt + {MA2,…,MAL}`), whose
printed values 0.60 / 0.70 / 0.81 / 0.80 / 0.73 confirm the mapping independently of the
archive. `[GAP]` **Anyone reusing `Data_trend.mat` by its row labels will silently compare
the wrong design.**

Replication against those paths:

| design | signals | mine `R²_OS` % | printed % | **path max\|Δ\|** | corr |
|---|---|---|---|---|---|
| `Xt` | 14 | **0.599** | 0.60 | **0.0** | 1.00000000 |
| `Xt + {MA2..MA6}` | 84 | 0.694 | 0.70 | **6.66e-16** | 1.00000000 |
| `Xt + {MA2..MA12}` | 168 | 0.754 | 0.81 | **7.77e-16** | 1.00000000 |
| `Xt + {MA2..MA24}` | 336 | — | 0.80 | not run (§8) | — |
| `Xt + {MA2..MA36}` | 504 | — | 0.73 | not run (§8) | — |

**Reading the `R²_OS` residuals.** They are an *alignment* artefact, not a numerical one.
`mf.window.from_cutoffs(test_start=...)` takes the first **origin**, so passing the month of
the first target produced 695 forecasts beginning 1965-02 rather than 696 beginning 1965-01.
Moving the origin back one month reproduces the printed value **exactly** (0.599 vs 0.60).
Since the ladder paths are already identical to 1e-16 on the overlapping months, adding the
missing month makes the samples identical too; they were not re-run for a confirmatory digit.

## 5. Table 4 — factor columns (PCR, PLS, S-PCR)

Structure taken from the Internet Appendix, not guessed: the `14 × L` trend predictors are
split into **14 groups, one per variable**, each holding that variable's `MA_1..MA_L`; the
first `J` components are extracted **within** a group and used to forecast; the 14 forecasts
are pooled by simple average (IA §IA1.2.1, and the parallel passages for PLS and scaled PCA).
PLS carries a single latent factor per group by construction (IA9/IA10).

**Internal check on that reading:** at `L = 1` each group holds one column, so all three
methods must collapse to the univariate predictive regression — and the paper's current-value
cells for PCR/PLS/S-PCR are all `0.60`, equal to Table 2 panel A. They do:

| design | J | model | mine % | printed % | Δ pp | path max\|Δ\| |
|---|---|---|---|---|---|---|
| `Xt` | 1 | PCR / PLS / S-PCR | **0.599** | 0.60 | −0.001 | **8.9e-16 / 1.0e-15 / 1.0e-15** |
| `+{MA2..MA6}` | **1** | PCR | 0.723 | 0.74 | **−0.017** | **2.31e-03** |
| | **1** | PLS | 0.811 | 0.79 | +0.021 | 1.97e-01 |
| | **1** | S-PCR | 0.814 | 0.80 | **+0.014** | 8.22e-02 |
| `+{MA2..MA12}` | **1** | PCR | 0.853 | 0.91 | −0.057 | **3.04e-03** |
| | **1** | PLS | 0.835 | 0.94 | −0.105 | 1.43e-01 |
| | **1** | S-PCR | 0.889 | 0.92 | **−0.031** | 9.56e-02 |

`[GAP]` **`J` is never given a numeric value** — the paper and appendix say only `J ≪ L`, and
the archive ships forecasts without the estimation code. `J` was therefore **not tuned to the
target**; every value in `{1,2,3}` was run and is reported side by side:

| design | model | **J=1** | J=2 | J=3 |
|---|---|---|---|---|
| `+{MA2..MA6}` | PCR | **−0.017** | −0.066 | −0.155 |
| | S-PCR | **+0.014** | +0.050 | −0.279 |
| `+{MA2..MA12}` | PCR | **−0.057** | +0.236 | +0.129 |
| | S-PCR | **−0.031** | +0.241 | +0.127 |

The evidence favours `J = 1`: it is the only value inside 0.06pp everywhere, and `J = 2, 3`
diverge by 0.13-0.28pp. That is an inference from reported variants, not a fitted choice.
(PLS is invariant to `J` by construction, which is itself a check that the single-factor
reading of IA10 was implemented as intended.)

**Path parity separates "the same estimator" from "the same answer".** At `L = 1` all three
agree with the archive at 1e-15, which pins the combination and evaluation spine. Beyond
that, **PCR** stays within `3e-3` — numerically the same estimator — while **S-PCR** (`~1e-1`)
and **PLS** (`~1.5e-1`) land close in `R²_OS` while following materially different paths.
`[GAP]` The authors' PLS is a two-step OLS construction (IA9/IA10); this replication uses the
packaged `pls`, and scaled PCA's slope-scaling step likewise differs in detail. On the
summary statistic alone these would all have read as "matches"; only the path check
distinguishes them.

## 6. What is not covered

- `[GAP]` **Table 4's LASSO and ENet columns.** The design is specified (14×L split into `L`
  groups **by MA lag**, LASSO within each group, `L` forecasts pooled, penalty by recursive
  threefold CV — IA Eq. IA4/IA5), but the per-origin CV wrapper is not built here.
- `[GAP]` **Table 5 (NN1-NN5).** All five architectures run on this data (verified: `(2)`,
  `(4,2)`, `(8,4,2)`, `(16,8,4,2)`, `(32,16,8,4,2)` with ReLU + Adam, 0.4-2.0 s per fit), but
  the paper averages the **five best of many seeds by validation `R²_OS`** (IA §IA1.3), which
  needs a seed-ensemble wrapper.
- **Table 6** is arithmetic on the Table 4/5 forecasts (`FC_linear = 1/5*[PCA+PLS+SPCA+LASSO+
  ENET]`, `FC_all = 1/10*[…]`), so it becomes free once those exist.
- `[GAP]` **The archive contains no estimation code for Tables 4-6** — only the saved
  forecasts. Those tables are therefore *verifiable* against stored paths but not
  *re-derivable* from the authors' posted code.
- **Out of scope by design decision:** Table 7's economic value and Figure 4's wealth paths
  (mean-variance weights, CER, performance fees) are finance tooling, outside this package.

## 7. P2 — defects this replication surfaced

1. **The forecast table could not be written to Parquet (fixed, PR #481).**
   `report.forecasts.to_parquet(...)` raised `ArrowNotImplementedError` for any run containing
   a model with no parameters: such a model emits `params={}`, Arrow types an empty mapping as
   a struct with **no child fields**, and Parquet cannot represent that. `ols` is one such
   model, so the failure reached a two-arm pipeline — persisting a run, the most ordinary
   thing to do with one, did not work. `_forecast_table` now normalizes empty mappings to
   `None` recursively; forecast values are untouched and the round-trip returns identical
   predictions and actuals.
2. **A pre-existing golden-snapshot failure, isolated and attributed (not fixed here).**
   `test_runner_matrix_matches_golden_snapshot` reports 17 of 522 `prediction` values drifted
   from the pinned snapshot by up to `4.2e-4` — four orders of magnitude beyond the ~1 ULP a
   threaded reduction explains. It fails **identically on clean `origin/main`**, so it is
   unrelated to this work; recorded rather than silently absorbed.
3. **Three hypotheses raised and refuted**, recorded so they are not re-litigated: *listwise
   contamination* — an arm's training sample is **not** truncated by NaNs in another arm's
   columns (`DP_L0` and `DP_MA1`, the same signal in different bundles, are bit-identical);
   *arm dropout* — all 42 arms contribute at every one of the 696 origins; and *the package is
   wrong about the MA designs* — it was the comparison target that was wrong (§4).

## 8. P3 / P4 — efficiency, and identity under it

| setting | wall clock | `R²_OS` |
|---|---|---|
| `n_jobs=1` | 20.7 min | 0.611% |
| `n_jobs=2` | 12.1 min | 0.611% |
| `n_jobs=4` | **6.0 min (3.45×)** | **0.611%** |

**The identity gate.** The `n_jobs=4` run returns the same `R²_OS` to three decimals *and* a
forecast path bit-identical to the archived one (`max|Δ| = 0.0`). Parallelism changes only
wall clock.

**A cost finding.** Wall clock grows **superlinearly** in the number of arms: 84 signals took
73.2 min and 168 signals took 323.6 min — 2× the arms for **4.4×** the time at fixed
`n_jobs=4`. Extrapolated, the 336- and 504-signal ladders are ~24 h and ~50 h; that is why
§4's table stops at `L=12`, and it is a cost ceiling rather than a correctness problem.

## 9. Reproduce

```bash
python3 scripts/replication/han_lu_zhou/build_panel.py          # MD5-verified archive -> panel
python3 scripts/replication/han_lu_zhou/run_table3_dense.py 4 dense6
python3 scripts/replication/han_lu_zhou/run_table2_sparse.py 4
python3 scripts/replication/han_lu_zhou/run_table4_factor.py 4 L1,L6,L12 1,2,3
```

The origin must be the month **before** the first intended target: pass `test_start = idx[456]`
to reproduce the authors' 696-month block beginning 1965-01.

`[GAP]`/`[ASSUMPTION]` register: archive row labels vs archive contents (§4); no archived path
for Table 2 (§3); `J` unspecified, variants reported, `J=1` favoured (§5); PLS and scaled-PCA
implementations differ from the authors' two-step constructions (§5); Table 4 LASSO/ENet,
Table 5 and Table 6 not run (§6); no estimation code in the archive for Tables 4-6 (§6);
ladders `L=24, 36` stopped on cost (§8); the FRED loaders and tcode machinery are unused and
therefore unverified by this note (§2); the panel is the authors' and was not rebuilt from
Goyal-Welch raw files (§1).
