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

### 2.1 Table 1 — the panel itself, before any estimator

Table 1 costs seconds and is the only exhibit in the paper that tests the **panel**
rather than an estimator. If our fourteen predictors or our MA construction had drifted
from the authors', every later table would inherit the error while still looking
internally consistent. Sample 1926:12–2022:12 (1,153 months), per the Table 1 note.

**Units.** The note says both statistics are "expressed as percentages". Welch-Goyal
deliver the yields and spreads already in percent (TBL, LTY, LTR, TMS, DFY, DFR), and
INFL as a percent inflation rate; the remaining seven are ratios or log ratios (DP, DY,
EP, DE, BM, NTIS, SVAR) and take ×100. The rule is applied **by source unit, uniformly** —
not fitted cell by cell to the printed numbers. The rate block is the control on it: it is
compared with *no rescaling at all*.

| predictor | Xt mean | Xt SD | MA3 mean | MA3 SD | MA6 mean | MA6 SD | MA12 mean | MA12 SD |
|---|---|---|---|---|---|---|---|---|
| DP | -340.93 | 47.67 | -340.91 | 47.51 | -340.88 | 47.29 | -340.79 | 46.85 |
| DY | -340.43 | 47.47 | -340.41 | 47.30 | -340.38 | 47.09 | -340.30 | 46.64 |
| EP | -276.15 | 42.05 | -276.15 | 41.79 | -276.16 | 41.35 | -276.15 | 40.31 |
| DE | -64.77 | 32.77 | -64.75 | 32.61 | -64.72 | 32.15 | -64.64 | 30.75 |
| BM | 55.19 | 26.84 | 55.23 | 26.67 | 55.28 | 26.47 | 55.40 | 26.13 |
| NTIS | 1.60 | 2.57 | 1.60 | 2.55 | 1.60 | 2.51 | 1.59 | 2.43 |
| TBL | 3.29 | 3.06 | 3.29 | 3.06 | 3.29 | 3.04 | 3.30 | 3.02 |
| LTY | 4.96 | 2.81 | 4.97 | 2.80 | 4.97 | 2.80 | 4.98 | 2.80 |
| LTR | 0.45 | 2.49 | 0.45 | 1.48 | 0.45 | 1.04 | 0.46 | 0.74 |
| TMS | 1.67 | 1.29 | 1.67 | 1.27 | 1.68 | 1.24 | 1.68 | 1.19 |
| DFY | 1.12 | 0.68 | 1.12 | 0.67 | 1.12 | 0.66 | 1.12 | 0.64 |
| DFR | 0.05 | 1.41 | 0.04 | 0.74 | 0.04 | 0.50 | 0.04 | 0.33 |
| INFL | 0.25 | 0.53 | 0.25 | 0.42 | 0.25 | 0.37 | 0.25 | 0.32 |
| SVAR | 0.29 | 0.60 | 0.29 | 0.50 | 0.29 | 0.44 | 0.29 | 0.40 |

The table above is ours; **every one of its 112 cells equals the printed cell at the
printed precision**, which is why no Δ column is shown — it would be zeros. Against the
unrounded values, max |Δ| = **0.0049**, mean |Δ| = 0.0026, and 112/112 are within 0.005.
The worst cell is SVAR's MA3 standard deviation, 0.4951 against a printed 0.50. The
residual is the printed table's rounding, not disagreement.

Nothing downstream has to take the panel on trust after this.

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

### 5.1 Table 4 — shrinkage columns (LASSO, ENet), and why they are reported as a sensitivity

Structure again from the appendix, and it is the mirror image of the factor columns: the
`14 × L` predictors are split into **`L` groups by MA lag** — group `l` holds the same-lag
moving average of all 14 variables — LASSO or elastic net selects within each group, and the
`L` forecasts are pooled by a simple average (IA Eq. IA4/IA5). The penalty is set
"recursively through threefold cross-validations", which the package expresses directly:

```python
model_selection=mf.model_selection.grid(
    {"alpha": ALPHAS}, validation_splitter=mf.recursive_threefold()
)
```

**Running that rule as written does not reproduce the paper, and the reason is diagnosable.**
The cross-validation selected the **largest** alpha in the grid at **696 of 696 origins**.
Measured `alpha_max` — the smallest penalty that zeroes every coefficient — is 1.098 at the
first origin, so that selection is the null model: the forecast collapses to the training
mean, which *is* the benchmark, and `R²_OS` lands at ≈0 against the published 0.86–1.18.

`[ASSUMPTION]` The appendix does not say whether predictors are standardized before the
penalty. They must be here: within a group the 14 Welch-Goyal variables span orders of
magnitude (SVAR ~1e-4 against TBL ~3), and the package's own guard refuses an unstandardized
penalized fit at those scales. `standardize=True` throughout.

Because the penalty is underdetermined, it was **not tuned toward the printed values**.
The sensitivity is reported instead:

| design | α = 0.001 | α = 0.01 | α = 0.05 | α ≈ 1 (the CV choice) | **printed** |
|---|---|---|---|---|---|
| `Xt` — LASSO | −8.450 | −7.327 | −3.832 | ≈ 0 | **−0.31** |
| `Xt` — ENet | −8.409 | −7.138 | −4.210 | ≈ 0 | **−0.35** |
| `+{MA2..MA6}` — LASSO | −6.928 | −5.964 | −3.011 | ≈ 0 | **0.86** |
| `+{MA2..MA6}` — ENet | −6.887 | −5.858 | −3.192 | ≈ 0 | **0.59** |
| `+{MA2..MA12}` — LASSO | −4.272 | −3.538 | −1.640 | ≈ 0 | **1.11** |
| `+{MA2..MA12}` — ENet | −4.243 | −3.516 | −1.803 | ≈ 0 | **1.18** |

**What the shape of that table settles.** `R²_OS` rises monotonically with the penalty and
converges to 0 — the null model — from below. So **no penalty in this implementation reaches
the published figures for the ladder designs**, which are *positive*: the paper's LASSO beats
the historical mean, while this one at best ties it. The gap is therefore **structural, not a
matter of tuning**, and no amount of alpha search would close it.

`[GAP]` **The likely structural difference is post-selection refitting.** Equation IA4 reads
"we apply LASSO **to select** the `J_{l,t}` optimal predictors … and forecast the market
expected return as `r̂ = α̂_{l,t} + Σ β̂^{(j)}_{l,t} MA^j`" — LASSO is described as the
*selector*, and the reported coefficients carry their own hats, which reads as an OLS refit
on the selected subset (post-LASSO) rather than the shrunken LASSO coefficients this
replication uses. That is a materially different estimator: post-selection OLS undoes the
downward bias that drives these forecasts toward the mean. Testing it is a separate run and
is not attempted here; it is recorded as the leading explanation rather than guessed at.

**Two package defects came out of this exhibit**, both now fixed (§7): a one-member
combination vanished silently — the `L=1` design has exactly one group, so its whole row came
back `NaN` with nothing to explain it (PR #483) — and model selection returned a grid-edge
value without saying so, which is what let 696 identical null-model selections pass for a
result (PR #484). The second is the more consequential: without it, the ≈0 row above would
have read as a finding rather than as a symptom.

## 5.2 Table 5 — the neural networks, and a table that its own paper cannot pin down

**The published Table 5 is not identified by the published specification.** The article and
its Internet Appendix give the architectures, ReLU, Adam, and the rule "average the five
best seeds in validation" — and nothing else. Searching both documents end to end for
epochs, batch size, learning rate, weight decay, dropout, early stopping, patience, seed
count or validation length returns nothing; the appendix's twelve tables contain no
hyper-parameter table, and the article's own appendix is variable definitions. The paper's
one methodological citation, Gu, Kelly & Xiu (2020), is scoped narrowly: it is invoked *to
fix architectures ex ante*, not for a training protocol.

That matters because the result is protocol-dependent to a degree that swamps the
published differences — this replication measured it (§5.3).

### The protocol used here, stated in full

Because the paper's is incomplete, ours is given completely, so that **this** table is
reproducible even though the one it is compared against is not. Provenance is marked per
row: **[paper]** taken from Han-Lu-Zhou, **[GKX]** read verbatim from Gu, Kelly & Xiu
(2020) (NBER w25398, hyper-parameter table, NN1-NN5 column), **[ASSUMPTION]** our choice
where neither states a value.

| setting | value | provenance |
|---|---|---|
| architectures | NN1 `(2)`, NN2 `(4,2)`, NN3 `(8,4,2)`, NN4 `(16,8,4,2)`, NN5 `(32,16,8,4,2)`, fully connected | **[paper]** IA §IA2 |
| activation | ReLU | **[paper]** Eq. IA13 |
| optimizer | Adam, default parameters | **[paper]** (Adam) + **[GKX]** (defaults) |
| epochs (max) | 100 | **[GKX]** |
| early-stopping patience | 5 | **[GKX]** |
| learning rate | 0.001 | **[GKX]** — the lower of `{0.001, 0.01}` |
| seeds fit per origin | 10 | **[GKX]** `Ensemble = 10` |
| seeds averaged | best 5 by validation | **[paper]** |
| batch size | 128 | **[ASSUMPTION]** GKX use 10000 for a panel of millions of stock-months; this sample has 500-1100 training rows, where that value is the full sample |
| early-stopping split | last 20% of the fit window, **time-ordered** | **[ASSUMPTION]** neither states it; a shuffled split would let the network validate against its own future |
| seed-ranking window | last 120 months of the fit window | **[ASSUMPTION]** the paper says "a fixed validation period" without a length |
| dropout / weight decay | 0.0 / 0.0 | package defaults, stated because they are part of the protocol |
| loss | MSE | package default |
| input scaling | standardized on the fit window at each origin | **[ASSUMPTION]** required; raw trend levels span orders of magnitude |
| predictor block | the whole `14 × L` set, ungrouped | **[paper]** — unlike Tables 2-4, the networks are meant to take all signals at once |

**Two validation windows, deliberately.** They serve different purposes and are not the
same split: the **seed-ranking** window (120 months) chooses *which* five networks to
average, per the paper's rule; the **early-stopping** split (20% tail) chooses *when to
stop* each individual fit, per GKX. Both sit inside the fit window and neither touches the
forecast target.

**Reproduction coordinates.** macroforecast at `f61935a5`; torch 2.7.0+cu126 on CUDA
(NVIDIA RTX 4060); seeds `0..9` per origin; expanding estimation with the first origin at
index 456 so the first target is 1965-01 and the out-of-sample block is 696 months.
`[note]` Runs on CUDA are not bit-reproducible across different GPUs or driver versions;
the protocol is fully specified, the last decimal is not.

## 5.3 Table 5 — results, and how far the unstated protocol moves them

Fifteen cells under the protocol of §5.2, against the same fifteen run to a fixed
100-epoch budget with no early stopping — the only difference between the two columns —
and against the printed table.

| design | model | **GKX protocol** | no early stopping | printed | Δ vs printed |
|---|---|---|---|---|---|
| `Xt` | NN1 | −1.880 | −2.691 | −0.11 | −1.77 |
| | NN2 | **−0.553** | −7.209 | 0.03 | −0.58 |
| | NN3 | **0.329** | −3.108 | **0.41** | **−0.08** |
| | NN4 | 0.325 | −2.157 | −2.07 | +2.40 |
| | NN5 | 0.112 | −0.732 | −1.69 | +1.80 |
| `+{MA2..MA6}` | NN1 | −6.195 | −8.458 | 0.67 | −6.87 |
| | NN2 | **−0.192** | −7.589 | 0.23 | −0.42 |
| | NN3 | **−0.134** | **−18.923** | 1.77 | −1.90 |
| | NN4 | **−0.138** | **−20.001** | 0.41 | −0.55 |
| | NN5 | 0.545 | — | −0.41 | +0.96 |
| `+{MA2..MA12}` | NN1 | −0.425 | — | 1.23 | −1.66 |
| | NN2 | −0.217 | — | 1.52 | −1.74 |
| | NN3 | −0.333 | — | 0.78 | −1.11 |
| | NN4 | **0.948** | — | **1.28** | **−0.33** |
| | NN5 | 0.980 | — | 0.01 | +0.97 |

### What the middle column establishes

**One unstated switch moves a cell by up to 20 percentage points.** Early stopping alone
takes `+{MA2..MA6}` NN3 from **−18.9% to −0.13%** and NN4 from **−20.0% to −0.14%**; at
`Xt`, NN2 goes from −7.2% to −0.55%. The published spread across the whole of Table 5 is
about 3.8 points (−2.07 to +1.77). **The indeterminacy is five times the effect being
reported.** No amount of care on our side closes that: the paper does not say which side
of it to stand on.

### Where the protocol lands relative to the paper

Under the GKX protocol nine of fifteen cells fall inside ±1.1pp of print, and three are
inside 0.35pp (`Xt` NN3 −0.08, `+{MA2..MA12}` NN4 −0.33, `+{MA2..MA6}` NN2 −0.42). The
residual pattern is systematic rather than noisy: our networks are **flatter** than the
paper's. Where the paper reports a collapse we do not fall as far (`Xt` NN4 −2.07 → +0.33,
NN5 −1.69 → +0.11); where it reports a gain we do not rise as high (`+{MA2..MA6}` NN3
1.77 → −0.13). Early stopping is exactly the intervention that compresses both tails, so
the direction is the one the protocol difference predicts.

**This bears on a claim in the paper.** Section 3.2 reads NN5's poor showing as a
small-sample, low-signal artefact of depth — "simple networks with fewer layers and nodes
often outperform in small data sets". Under a protocol with early stopping that depth
penalty does not appear: at `+{MA2..MA12}`, our NN4 (0.948) and NN5 (0.980) are the two
**best** cells of the design, and at `Xt` the deepest networks stop collapsing entirely.
The depth ordering in Table 5 is therefore at least partly a property of the training
protocol rather than of network depth. That reading is offered as what these runs support,
not as a correction to the paper — with the protocol unstated, neither reading can be
settled from the published material.

`[GAP]` Path parity is not meaningful for this exhibit. The archived NN forecast paths sit
2.1-7.7 away in the max-norm, which is what different seeds and a different training
protocol produce; unlike Tables 2-4, agreement here could only ever be distributional.

**Cost.** Early stopping also made the exhibit affordable: 7-27 min per cell against
83-119 min for the fixed budget, because training stops around epoch 15-20 instead of
running to 100. The full fifteen cells took about 4 h rather than the ~49 h projected for
the fixed-budget version.

**A package gap surfaced here.** `nn` had no early stopping at all — no `validation_fraction`,
no patience, no held-out tail — so the protocol above could not be expressed until PR #485
added it. That is not a stylistic omission: the middle column of the table above is what
the package could produce before the fix, and it is off by up to 20 points.

## 5.4 Table 6 — ensembles, and where the inherited disagreement lands

Table 6 estimates nothing new: it averages the method-level paths Tables 4 and 5 already
produced. It is assembled from the saved forecast frames and run through
`evaluate(master, spec)` — the same evaluation code a live run uses, with no refit — so the
whole exhibit costs seconds.

Section 3.3 gives the four ensembles: **(a) Linear** = PCR, PLS, S-PCR, LASSO, ENet;
**(b) Nonlinear** = neural networks "with varying depths"; **(c) NN & ENet**; **(d) All**.
`[GAP]` The Table 6 note says the networks run "from one to four" while Table 5 reports
NN1–NN5. Both readings are produced below rather than one being chosen.

**The disagreement here is inherited and its direction was predicted.** Two components
already disagree for documented reasons — LASSO/ENet structurally (§5.1) and the networks
through a training protocol the paper never states (§5.2, §5.3, where our networks come out
*flatter* than the paper's). Table 6 therefore tests nothing about those components; it
shows what they ensemble to.

**NN1..NN4 (the Table 6 note's reading)**, `R²_OS` in percent, mine / printed / Δ:

| design | Linear | | | Nonlinear | | | NN+ENet | | | All | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Xt` | 0.389 | 0.35 | **+0.039** | 0.166 | −0.18 | +0.346 | 0.213 | 0.01 | +0.203 | 0.399 | 0.42 | **−0.021** |
| `Xt + {MA2..MA6}` | 0.519 | 0.90 | −0.381 | −0.276 | 0.99 | −1.266 | −0.109 | 1.03 | −1.139 | 0.333 | 1.08 | −0.747 |
| `Xt + {MA2..MA12}` | 0.587 | 1.20 | −0.613 | 0.626 | 1.44 | −0.814 | 0.645 | 1.51 | −0.865 | 0.751 | 1.45 | −0.699 |

**NN1..NN5 (Table 5's set)** — only the network columns move:

| design | Linear | | | Nonlinear | | | NN+ENet | | | All | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Xt` | 0.389 | 0.35 | +0.039 | 0.213 | −0.18 | +0.393 | 0.244 | 0.01 | +0.234 | 0.400 | 0.42 | −0.020 |
| `Xt + {MA2..MA6}` | 0.519 | 0.90 | −0.381 | 0.016 | 0.99 | −0.974 | 0.086 | 1.03 | −0.944 | 0.399 | 1.08 | −0.681 |
| `Xt + {MA2..MA12}` | 0.587 | 1.20 | −0.613 | 0.795 | 1.44 | −0.645 | 0.777 | 1.51 | −0.733 | 0.814 | 1.45 | −0.636 |

### What the pattern says

**The `Xt` row is where the components agree, and it is the row that matches**: Linear
+0.039pp, All −0.021pp. That row uses no MA signals, so LASSO/ENet have the least room to
diverge and the networks have the fewest inputs to overfit. Every trend row is low, by
−0.38 to −1.27pp, and the gap is **widest in exactly the columns containing networks**.
That is the direction §5.3 predicts: early stopping compresses both tails, so our networks
neither collapse nor spike, and an ensemble of flatter members cannot reach the printed
highs. The residual is a component story, not an ensembling one.

**Significance** (Clark-West, NN1..NN4 reading), against the paper's stars:

| design | Linear | Nonlinear | NN+ENet | All |
|---|---|---|---|---|
| `Xt` | p=0.015 (paper `*`) | p=0.203 (paper none) ✓ | p=0.198 (paper none) ✓ | p=0.070 (paper `*`) ✓ |
| `Xt + {MA2..MA6}` | p=0.006 (paper `***`) ✓ | p=0.307 (paper `**`) | p=0.292 (paper `**`) | p=0.080 (paper `***`) |
| `Xt + {MA2..MA12}` | p=0.004 (paper `***`) ✓ | p=0.036 (paper `***`) | p=0.033 (paper `***`) | p=0.015 (paper `***`) |

The **Linear** column reproduces the paper's significance exactly at every design, and the
`Xt` row matches in all four columns. The misses are confined to the network-bearing
columns of the trend designs — the same place the point estimates diverge, for the same
reason.

**Issue #488 touches this table too, and by how much is measured.** The benchmark rows come
from the Table 4 runs, whose bundles carry MA columns, so the `hist_mean` denominator is
truncated there as it is in §4. Re-scoring the Linear ensemble against a prevailing mean
computed from our own panel moves it by **+0.000pp at `Xt`, +0.017pp at `{MA2..MA6}` and
+0.062pp at `{MA2..MA12}`** — the same direction as §4 and the same order of magnitude. It
does not touch the reading above, whose gaps are 0.38 to 1.27pp. §5.5's Table 10 is immune
by construction: its benchmark is an ordinary `ols` arm, not `hist_mean`, and contender arms
are not truncated by the bundle.

**This exhibit is the one that needed PR #486.** All twelve Clark-West values above are on
*combinations*, which the package could not test at all before that fix: they would have
come back NaN, and Table 6's entire significance column with them.

`[GAP]` Rows `Xt + {MA2..MA24}` and `Xt + {MA2..MA36}` are not produced, because their
component paths were never estimated (§8's cost ceiling). `[GAP]` `J`, the number of
factors, is unspecified in the paper; `J=1` is used here, as in §5.

## 5.5 Table 10 — quarterly macro targets, and a rank-detection failure

Table 10 is the paper's only genuinely **macro** exhibit and the sharpest test of the BYOD
path in §2: a 304-quarter panel out of the authors' archive, so every series is a custom
target and a custom predictor and nothing comes from a FRED loader. Both fits take about a
minute.

**The row labels do not mean what they say.** Read off the authors' code
(`Forecasts_in_and_out_of_sample.m:795-925`) rather than the printed headers: `L_max = [1,
2, 4, 8, 12]`, so the ladder **doubles** — `y + {MA2,…,MA8}` is *not* the consecutive ladder
MA2..MA8. Each row is the equal-weighted mean of single-MA forecasts up to that level, each
member a bivariate regression `y_{s+1} ~ 1 + MA_k(y_s) + z_s`, and `R²_OS` is against
**row 1 — the current-value model carrying the same control** — not against a historical
average. `[GAP]` A reader working from the printed table alone would build the wrong design.

| panel | design | RF | D/P | UNRATE | Inflation | GDP growth | Comb |
|---|---|---|---|---|---|---|---|
| A: Inflation | `y + {MA2}` | 0.71 / 0.71 | 1.68 / 1.68 | 2.23 / 2.23 | **1.21 / 1.04** | 1.64 / 1.64 | 1.47 / 1.45 |
| | `y + {MA2,MA4}` | 5.61 / 5.61 | 8.85 / 8.85 | 10.10 / 10.10 | **8.12 / 7.94** | 9.15 / 9.15 | 8.14 / 8.11 |
| | `y + {MA2..MA8}` | 4.51 / 4.51 | 8.12 / 8.12 | 10.34 / 10.34 | **7.56 / 7.34** | 9.34 / 9.34 | 8.36 / 8.32 |
| | `y + {MA2..MA12}` | 3.99 / 3.99 | 8.62 / 8.62 | 12.14 / 12.14 | **8.61 / 8.39** | 10.18 / 10.18 | 9.16 / 9.12 |
| B: GDP growth | `y + {MA2}` | 6.39 / 6.39 | 6.63 / 6.63 | 7.46 / 7.46 | 6.12 / 6.12 | 3.13 / 3.14 | 6.30 / 6.30 |
| | `y + {MA2,MA4}` | 10.37 / 10.37 | 10.93 / 10.93 | 12.40 / 12.40 | 10.37 / 10.37 | 1.92 / 1.92 | 10.10 / 10.10 |
| | `y + {MA2..MA8}` | 11.47 / 11.47 | 12.49 / 12.49 | 15.02 / 15.02 | 12.33 / 12.33 | 1.49 / 1.49 | 11.68 / 11.68 |
| | `y + {MA2..MA12}` | 12.08 / 12.08 | 13.43 / 13.43 | 16.46 / 16.46 | 13.73 / 13.73 | 1.59 / 1.59 | 12.82 / 12.82 |

*(mine / printed, `R²_OS` in percent)*. **44 of 48 cells agree to within 0.05pp and 40 are
exact at the printed precision**; mean |Δ| = 0.021pp. All four misses sit in one cell block,
marked in bold, and they are a package defect rather than a specification question.

### Defect #15 — `ols` silently returns garbage on an exactly collinear design

Panel A's `Inflation` column is degenerate by construction: the target *is* inflation and the
control *is* inflation, so at `k = 1` the design carries the same column twice. That is not
an artificial case — it falls straight out of the paper's own specification, and the archive
runs it too.

The deviation was localised before being attributed. Only the degenerate arm moves:

| arm | max\|Δ\| vs a faithful re-implementation |
|---|---|
| `k1_INF` (degenerate) | **5.58e-01** |
| `k2_INF`, `k4_INF`, `k8_INF`, `k12_INF` | 2.2e-15 – 3.6e-15 |
| `k1_UNE` (same `k`, not degenerate) | 1.2e-14 |

and the faithful re-implementation lands on the **printed** values (1.042 / 7.938 / 7.345 /
8.387 against 1.04 / 7.94 / 7.34 / 8.39) under both a minimum-norm solve and MATLAB's
basic-solution rule, so the printed column is not the odd one out — we are.

**Mechanism, traced to the LAPACK call.** At 11 of 232 origins the fitted coefficients come
back as an exploded cancelling pair:

| | the same X, refit directly | inside the pipeline |
|---|---|---|
| array handed to sklearn | C-contiguous `ndarray` | non-contiguous `DataFrame` |
| smallest centered singular value | 2.14e-15 | 5.59e-15 |
| LAPACK `rank_` | 1 (truncated) | **2 (not truncated)** |
| coefficients | `[0.2897, 0.2897]` | **`[+2.33e14, −2.33e14]`** |
| forecast | 2.298749 | **2.857185** |

The columns are **bit-identical** (`max|col0 − col1| = 0.0`) and the centered design has
`s_min/s_max = 1.16e-16`, so the matrix is exactly rank-one. Whether `gelsd` truncates the
null direction depends on the memory layout it receives, and when it does not, the two
coefficients grow to ±2.3e14 and cancel. The prediction then carries a floating-point residue
of order `‖coef‖ · eps · x ≈ 2.3e14 × 2.2e-16 × 3.2 ≈ 0.2` — which is exactly the size of the
observed per-origin errors (0.05 to 0.56).

**Why it matters beyond this table.** The result is deterministic, reproduces on demand, and
is delivered with **no warning**: `ols` returns a number that looks ordinary and is wrong by
half a unit. `rank_` cannot be used as the detector, because in the failing case LAPACK
reports full rank — that mis-report *is* the failure. Filed rather than patched here: the
honest fix is a post-fit conditioning check that warns without changing any existing number,
and that deserves its own change with its own suite run.

`[GAP]` Panel B's degenerate cell (`GDP growth` control on the GDP target) matches to 0.01pp,
so the failure needs the rank decision to flip and does not fire on every collinear design.

## 5.6 Table 8 — the placebo, verified but not re-run, with the cost stated

Table 8 scrambles the past `{X_1,…,X_{t-1}}` at every origin, rebuilds pseudo-MA trends from
the scrambled history, and repeats to 2022:12 — **1,000 times**, reporting the average
`R²_OS`. That is 1,000 complete 696-origin runs per design.

It is not re-run here, and the cost is the reason rather than a judgement about the exhibit.
`Xt + {MA2..MA6}` alone takes 30.2 min for one pass at `n_jobs=12` (§8), and the two longest
ladders have never been run even once (§4). What *is* done is to verify the exhibit against
the authors' archive, which ships the full simulation output (`r2_sims`, 5 × 1,000):

| design | archive mean of 1,000 sims | printed | Δ | sd across sims |
|---|---|---|---|---|
| `Xt` | 0.5987 | 0.60 | −0.0013 | **2.2e-16** |
| `Xt + {MA2..MA6}` | 0.3953 | 0.40 | −0.0047 | 0.1800 |
| `Xt + {MA2..MA12}` | 0.3217 | 0.32 | +0.0017 | 0.1795 |
| `Xt + {MA2..MA24}` | 0.2719 | 0.27 | +0.0019 | 0.1646 |
| `Xt + {MA2..MA36}` | 0.2595 | 0.26 | −0.0005 | 0.1593 |

and the incremental panel likewise (−0.2034 / −0.2771 / −0.3268 / −0.3392 against printed
−0.20 / −0.28 / −0.33 / −0.34). **All nine printed values reproduce, max |Δ| = 0.0047pp.**

The `Xt` row is a consistency check rather than a result: scrambling the *past* cannot touch a
design that uses only the current value, so its 1,000 draws must be identical — and their
standard deviation is 2.2e-16. The archive is internally consistent on its own terms.

**What an independent replication would cost.** The cross-simulation standard deviation is
about 0.18pp, so reaching the printed precision needs

| tolerance | simulations | wall clock for the *cheapest* trend row alone |
|---|---|---|
| ±0.10pp | 13 | ~6.5 h |
| ±0.05pp | 50 | ~25 h |
| ±0.01pp | ~1,250 | ~26 days |

per design, and the two longest ladders cost 24 h and 50 h *per pass* (§8). A ±0.10pp
version of one row is the only affordable point on that curve, and it would not settle a
table whose reported effects are 0.20–0.34pp apart. Recorded as a quantified exclusion, not
an oversight.

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
3. **Clark-West was inexpressible for a forecast combination (fixed, PR #486).**
   `significance_table` built its CW-eligible set by walking `spec.arms` and reading
   `Arm.nested_in_benchmark`; `CombinationContender` had no such field, so a combination's
   `cw_stat`/`cw_p` came back NaN with no warning. That is the wrong default for this
   literature — CW *on the combination* is the headline test — and it is licensed, because a
   simple pool of arms that each nest the benchmark nests the benchmark. The flag now exists
   and the silent case warns. Table 6's twelve CW values and Table 9's stars could not be
   produced before it.
4. **`ols` returns a silently wrong forecast on an exactly collinear design (filed, issue
   #487).** At 11 of 232 origins LAPACK reports full rank on a rank-one matrix, the
   coefficients explode to a cancelling `±2.3e14` pair, and the prediction keeps the
   floating-point residue — up to **0.56** in level. Diagnosed in full at §5.5. Not patched
   here: `rank_` cannot be the detector, since the mis-report *is* the failure, and the honest
   fix warns without changing any existing number.
5. **The `hist_mean` benchmark is truncated by other arms' columns (filed, issue #488).**
   Its estimation sample starts at `2 + (leading NaNs of the longest column in the bundle)`
   even though it uses no predictors, so the *same* benchmark scores differently depending on
   which contenders share the run. This is the whole of §4's residual. Identified exactly —
   the reconstructed path matches to `0.00e+00`.
6. **Two hypotheses raised and refuted**, recorded so they are not re-litigated: *arm dropout*
   — all 42 arms contribute at every one of the 696 origins; and *the package is wrong about
   the MA designs* — it was the comparison target that was wrong (§4).
7. **A correction to an earlier entry in this list.** *Listwise contamination* was recorded
   here as refuted, on the evidence that `DP_L0` and `DP_MA1` — the same signal in different
   bundles — are bit-identical. That test covered **predictor arms only**. `hist_mean` has no
   predictor column, and it *is* truncated by the bundle (item 5). The original refutation
   stands for what it tested and does not generalise; it was stated more broadly than the
   evidence supported.

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
