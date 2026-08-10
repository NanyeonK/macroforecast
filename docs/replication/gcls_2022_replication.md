# GCLS (2022, JAE) Replication — macroforecast Trust Document

**Paper.** Goulet Coulombe, Leroux, Stevanovic, Surprenant (2022), *"How is Machine
Learning Useful for Macroeconomic Forecasting?"*, Journal of Applied Econometrics
37(5) (DOI 10.1002/jae.2924).

**Objective** (per `REPLICATION_OBJECTIVES.md`). (1) Reproduce the paper with
macroforecast as public trust evidence; (2) surface package bugs during
replication; (3) improve technical efficiency; (4) speedups that are
statistically identical (no draw/tolerance/tree reduction).

**Setup.** FRED-MD panel (official archive 2018-01 vintage), 5 targets
(INDPRO, CPIAUCSL, HOUST, UNRATE, T10YFFM≡SPREAD), horizons h∈{1,3,9,12,24},
456 pseudo-OOS origins (test 1980:01–2017:12, expanding estimation from 1960:01,
`retrain_every=1`, `retune_every=24`), 46 forecasting models (Table 1). Benchmark
= AR,BIC. All results below are pseudo-OOS, leak-free unless a cell is explicitly
labelled an author-surface diagnostic.

---

## Summary verdict

**GCLS 2022 replication: substantial but PROVISIONAL.**

INDPRO has been revalidated end to end under the corrected `X_t` specification and
the seeded splitter — 46 arms × 5 horizons, median |Δ rel-RMSPE| 0.0389, family
ordering reproduced. **The remaining four targets have not.** Their tables below are
still stacked-panel output from before the `lags` correction, and are labelled as
such wherever they appear.

So the accurate claim today is narrower than "verified faithful and correct", which
is what this section said until 2026-08-09. That phrasing described the INDPRO
result while the document also stated, a few sections down, that four of five
targets were superseded. A verdict cannot be broader than the run behind it.

**What is established:**

- INDPRO reproduces under the paper's specification, with the headline direction
  (ML/factor beats AR) intact and the least flattering number recorded: 104/225
  cells beat the AR,BIC benchmark against the paper's 128/225.
- The `lags` correction is supported by a *controlled* A/B — same commit, same day,
  one variable — with a `predictors=[]` control arm at `max |Δ| = 0`.
- The factor-extraction convention question is answered against the paper, on the
  120 deterministic cells, with a bit-identical noise floor as its control.
- Seven package defects were found and fixed along the way (Purpose 2), most
  recently the k-fold seed (#513 → #515).

**What is required before this reads "complete":**

| # | Remaining work | Status |
|---|---|---|
| 1 | 21 `,KF` arms (105 cells) re-run under post-#515 code | **done** — see 1a′ |
| 2 | `covpca_full` factor-convention grid | **done** — see 1a″ |
| 3 | CPIAUCSL / HOUST / UNRATE / T10YFFM re-run at `lags=0` | **in progress** |
| 4 | Pre-fix tables removed or moved to a labelled archive section | not started |

Until 3 and 4 land, the replication index should carry this paper as
**IN PROGRESS**, not as a completed trust result.

```{admonition} Correction (2026-08-07): the factor-estimation residual was OURS
:class: important

An earlier version of this verdict also attributed the residual to the paper's
"exact factor estimation". That was wrong, and the correction matters because it
moves a gap from *unexplainable* to *fixed*.

`registry.py::_far_features` called `mf.feature_spec(...)` without `lags`, and the
package default is `lags=(0, 1)`. Every predictor therefore entered at `t` and
`t-1`, so `far`'s internal PCA ran on a **stacked `[X_t, X_{t-1}]` panel — 264
columns for 132 series** — instead of the paper's cross-section
`X_t = Lambda F_t + u_t` (eq. swardi2). Measured directly by instrumenting
`MODEL_SPECS["far"].fit_func`: the PCA input columns were
`['RPI_lag0', 'RPI_lag1', 'W875RX1_lag0', 'W875RX1_lag1', ...]`.

Correcting it (`lags=0`) closes most of the gap on `ARDI,BIC`/INDPRO and, more
tellingly, **removes the horizon-growing pattern that motivated the
"under-specified" reading in the first place**:

| `ARDI,BIC` vs paper | stacked `[X_t, X_{t-1}]` | paper `X_t` |
|---|---|---|
| median abs deviation | 0.0856 | **0.0441** |
| h >= 9 | 0.1051 | **0.0459** |
| h = 24 | 0.1221 | **0.0182** |

**This table is a controlled A/B**, re-verified 2026-08-08: both columns were
produced on the same day from the same commit, with `MF_GCLS_XLAGS` the only
difference between the two runs. The re-run reproduced all six figures exactly.

Its negative control is `AR,BIC`, which is built with `predictors=[]` and therefore
cannot be reached by any predictor-lag setting:

```
AR,BIC across the two settings: max |delta| = 0.000e+00
```

Bit-identical, as it must be. That control also matters for reading the rest of this
document: `AR,BIC` is stable run to run, while the CV-selected arms are not (see the
reproducibility box below), so a controlled A/B is only meaningful on the IC-selected
and `,POOS` arms in the first place.

Still not a package defect — `lags=(0, 1)` is a reasonable general default and
the replication simply never stated the paper's design. But it was *our* spec
error, not the paper's silence, and the distinction is the whole point of a trust
note.

INDPRO has now been re-run end to end under `X_t` — see **1a′** below for the full
46-arm table, and for what that re-run does and does not establish. The four other
targets are still stacked-panel and are labelled as such wherever they appear.
```

| Source exhibit | Reproduced? | Note |
|---|---|---|
| **Table 1** (model list) | ✅ | all 46 models native |
| **Table A1 / RelMSPE tables** (App A) | ✅ substantially | headline ML>AR 25/25; RF family ~2pp; most families ≤5pp |
| **Table 2** (CV-selection coefficients) | △ direction only | collapse resolved; magnitude gap (K-fold long-horizon over-selection) |
| **Treatment-effect figures** (§"When are ML NL important") | ◑ partial | X + **RF-nonlinearity reproduced**; KRR-nonlinearity is a factor-precision gap |
| **Appendix B** (robustness of treatment effects) | ✅ (subsample) | ML gains concentrate in recessions/recent/long-h — reproduced. real-time/rolling: in progress |
| **Appendix C** (additional subsample graphs) | ✅ | last-20yr & long-horizon cuts reproduce (RF-NL +22.5 / +16.1) |
| **Appendix D** (NN + Boosted-Trees robustness) | ✅ | RF/BT/NN all show positive NL (+11.1 / +8.9 / +6.2) |
| Density (PIT), abs-loss, sign, quarterly, Canadian | not attempted | out of current scope |

---

## Exhibit 1 — GCLS-2022 Table A1 (Detailed Predictive Performance / RelMSPE tables)

The 46-model × 5-target × 5-horizon relative-RMSPE horse race. macroforecast values
are √(relative_mse) vs the AR,BIC benchmark, compared to the paper's published
rel-RMSPE (full sample).

**1a — Best model per target × horizon (mine vs paper).**

| Target | h | Mine best (rel-RMSPE) | Paper best (rel-RMSPE) | ML beats AR (both<1) |
|---|---|---|---|---|
| INDPRO | 1 | RFARDI,POOS (0.932) | B2,ridge,POOS (0.933) | yes |
| INDPRO | 3 | B3,ridge,KF (0.904) | B3,lasso,KF (0.913) | yes |
| INDPRO | 9 | B3,EN,POOS (0.969) | KRRARDI,POOS (0.921) | yes |
| INDPRO | 12 | RFARDI,POOS (0.949) | KRRAR,KF (0.910) | yes |
| INDPRO | 24 | KRRARDI,POOS (0.907) | RFARDI,KF (0.890) | yes |
| CPIAUCSL | 1 | B2,ridge,KF (0.904) | B1,EN,KF (0.908) | yes |
| CPIAUCSL | 3 | RFARDI,KF (0.907) | KRRAR,KF (0.888) | yes |
| CPIAUCSL | 9 | RFARDI,KF (0.834) | KRRAR,KF (0.836) | yes |
| CPIAUCSL | 12 | RFARDI,KF (0.849) | KRRAR,KF (0.827) | yes |
| CPIAUCSL | 24 | RFARDI,KF (0.833) | SVR-ARDI,RBF,KF (0.797) | yes |
| HOUST | 1 | B1,EN,KF (0.956) | ARDI,BIC (0.973) | yes |
| HOUST | 3 | RFARDI,POOS (0.960) | KRRARDI,POOS (0.943) | yes |
| HOUST | 9 | B1,lasso,POOS (0.947) | KRRARDI,POOS (0.915) | yes |
| HOUST | 12 | B1,lasso,POOS (0.961) | RFARDI,KF (0.914) | yes |
| HOUST | 24 | RFARDI,KF (0.917) | RFARDI,KF (0.838) | yes |
| UNRATE | 1 | B3,ridge,POOS (0.911) | B2,EN,POOS (0.907) | yes |
| UNRATE | 3 | B3,EN,KF (0.846) | SVR-ARDI,Lin,KF (0.873) | yes |
| UNRATE | 9 | B3,EN,POOS (0.893) | KRRARDI,KF (0.827) | yes |
| UNRATE | 12 | SVR-ARDI,RBF,KF (0.896) | KRRARDI,POOS (0.813) | yes |
| UNRATE | 24 | SVR-ARDI,RBF,KF (0.860) | RFARDI,POOS (0.763) | yes |
| T10YFFM | 1 | KRRARDI,KF (0.897) | RRARDI,POOS (0.936) | yes |
| T10YFFM | 3 | RFARDI,POOS (0.860) | RFARDI,POOS (0.830) | yes |
| T10YFFM | 9 | SVR-ARDI,Lin,POOS (0.889) | KRRAR,POOS (0.949) | yes |
| T10YFFM | 12 | SVR-ARDI,Lin,POOS (0.836) | KRRARDI,KF (0.827) | yes |
| T10YFFM | 24 | SVR-ARDI,Lin,POOS (0.830) | B1,lasso,POOS (0.844) | yes |

**Headline reproduced: ML/factor beats the AR benchmark in 25/25 cells.** Winning
*values* are near-ties with the paper (e.g. INDPRO h1 0.932 vs 0.933); the winning
*arm* often differs among near-tied nonlinear/factor models (exact-arm match 1/25).

```{admonition} Package bug found here: the CV-selected arms are not reproducible
:class: danger

Half the 46 arms select hyperparameters by 5-fold CV (`,KF`), and **those arms do
not reproduce run to run**. Found while trying to A/B the predictor-lag setting:
two runs with identical code, identical settings and identical `n_common` gave
different answers.

| run | `AR,BIC` rmse | `AR,KF` rmse |
|---|---|---|
| a (lags=0) | 0.002776 | 0.002791 |
| b (lags=0, repeat) | 0.002776 | **0.002781** |
| c (lags=0,1) | 0.002776 | 0.002791 |

```
a vs b (identical settings)  max |Δ rel-RMSPE| = 0.00387
a vs c (the setting tested)  max |Δ rel-RMSPE| = 0.00025
```

**The run-to-run spread is 16× the effect being measured.** `a` and `c`, which
differ in the setting under test, agree; `a` and `b`, which differ in nothing, do
not. `AR` is target-only (`predictors=[]`), so no predictor setting can reach it —
and `AR,BIC` is bit-stable across all three, so the deterministic selection path
is fine. The instability is specific to the CV path.

**Root cause found, and it is one default.** `random_kfold_split` seeds itself with
`random_state=0`, but `make_splitter` declared `random_state: int | None = None` and
forwarded that value *explicitly*, so the callee's seeded default was never reached.
Every `validation_splitter("random_kfold", ...)` goes through `make_splitter`, so
every k-fold shuffle in the package drew from OS entropy:

```
random_kfold_split(200, n_splits=5) twice       -> IDENTICAL
make_splitter("random_kfold", 200, n_splits=5)  -> DIFFERS
make_splitter(..., random_state=0) twice        -> IDENTICAL
```

Two hypotheses died on measurement first, and both are worth recording because each
looked right. It is **not** a parallel artifact: `n_jobs=1` is non-deterministic too
(7.4e-04), merely 5x smaller than `n_jobs=8` (3.9e-03). It is **not** floating-point
non-associativity in BLAS: `AR,BIC` is bit-identical to ten decimals across all four
runs, serial and parallel, and BIC selection issues the same `lstsq` calls — moved
numerics would have moved it too. Instrumenting `_resolve_selection_splits` then
showed the splits themselves differing between calls sharing one splitter spec.

The effect is not last-bit noise. Five `select_params` calls on identical inputs gave
`best_score` 1.00970 / 0.98571 / 1.00746 / 0.97873 / 1.00451, and AR order selection
over `{1,2,3,4,6,12}` picked **3, 4, 6, 3, 1** — a different model each time.

Fixed in PR #515 (four sites defaulted `None` and forwarded it; all now default to
`0`, matching `mf.window.random_kfold()`, which already promised a seed).

**Resolved: the table has been re-run under the fix.** The 46-arm × 5-horizon grid was
recomputed on a lane carrying the seed commit *alone* (cherry-picked rather than merged,
so nothing else moved with it). Every `,KF` number in this document is now a reproducible
value rather than one draw from a distribution.

**Fixing the seed did not improve parity, and there was no reason to expect it to.**

| INDPRO, 225 contender cells | median \|Δ rel-RMSPE\| | beats AR,BIC |
|---|---|---|
| unseeded | 0.0405 | 106 |
| **seeded (#513 fixed)** | **0.0389** | 104 |

Restricted to the 105 `,KF` cells the fix actually touches: 67 moved, median move 0.0017,
largest 0.0526 — and against the paper, 31 landed closer while 36 landed farther. That is
a wash, which is the correct outcome. An unseeded draw is not *biased*, only
*unrepeatable*; re-drawing with a fixed seed has no reason to move systematically toward
the paper. **Reproducibility and accuracy are different properties, and only the first was
broken.**

It also retro-justifies the caution in *"What the re-run does NOT establish"* below: part
of the cell-level churn between the stacked-panel and corrected runs was never
attributable to `lags` at all.

```{admonition} The control, and the one cell that failed it
:class: tip

The two runs differ by exactly one commit, so the 120 deterministic cells (`,POOS` +
`AIC`/`BIC`) had to come back identical. 119 do, five of them differing by a single ULP
(1.1e-16 — floating-point representation, not arithmetic).

One does not: `B3,lasso,POOS` at h=9 moved by **2.199e-05**. Recorded rather than rounded
away. It is not reachable by the seed — `lasso(X, y, *, alpha, max_iter, standardize)`
takes no `random_state` — and is consistent with a coordinate-descent solver stopping on
its tolerance at a marginally different iterate once a ULP-level difference appears
upstream. For scale, it is ~1,800× smaller than this table's median gap to the paper and
~180× smaller than the run-to-run noise it replaced.
```


Filed as macroforecast issue #513 (ledger `MF-001`) — Purpose 2, a package defect
surfaced by replication rather than by its own test suite.
```

**1a′ — corrected re-run, INDPRO, all 46 arms × 5 horizons (225 contender cells).**

The tables in 1a/1b/1c below were produced with the stacked `[X_t, X_{t-1}]` panel.
INDPRO has since been re-run end to end under the paper's `X_t` (456 origins, 46
arms, h∈{1,3,9,12,24}); these are those numbers. **The other four targets have not
been re-run and their rows in 1a/1c remain stacked-panel.**

| | median \|Δ rel-RMSPE\| | mean | n cells |
|---|---|---|---|
| **INDPRO, corrected `X_t`** | **0.0405** | 0.0587 | 225 |

By horizon:

| h | 1 | 3 | 9 | 12 | 24 |
|---|---|---|---|---|---|
| median \|Δ\| | 0.0307 | 0.0362 | 0.0703 | 0.0354 | 0.0436 |

By family:

| Model family | median \|Δ\| | n cells |
|---|---|---|
| RF (random forest) | **0.0194** | 20 |
| Linear (AR / ARDI, IC- and CV-selected) | 0.0247 | 35 |
| RR (reduced rank) | 0.0323 | 20 |
| Shrinkage (B1–B3) | 0.0478 | 90 |
| KRR (kernel ridge) | 0.0599 | 20 |
| SVR | 0.0883 | 40 |

Best arm per horizon, mine vs paper:

| h | Mine best | Paper best |
|---|---|---|
| 1 | RFARDI,POOS (0.932) | B2,ridge,POOS (0.933) |
| 3 | B3,ridge,KF (0.904) | B3,lasso,KF (0.913) |
| 9 | B3,EN,POOS (0.969) | KRRARDI,POOS (0.921) |
| 12 | RFARDI,POOS (0.949) | KRRAR,KF (0.910) |
| 24 | KRRARDI,POOS (0.907) | RFARDI,KF (0.890) |

The winning *value* stays a near-tie with the paper at h=1 (0.932 vs 0.933) and the
family ordering is reproduced — RF tightest, SVR loosest — but the winning *arm*
matches in 0/5 horizons, because the top handful of nonlinear/factor models are
separated by less than the residual gap.

One number is worth stating plainly because it is the least flattering: **106 of 225
cells beat the AR,BIC benchmark here, against 128 of 225 in the paper.** The headline
direction (ML/factor methods beat AR) reproduces; the *breadth* of that win does not
fully reproduce, and the shortfall sits in the SVR and shrinkage families.

### 1a″ — factor-extraction convention: covariance or correlation?

The corrected run leaves one internal inconsistency. `far` hard-codes **covariance**
PCA (centred only), while the B1–B3 shrinkage arms reach their factors through
`pca_step`, whose default is **correlation** (`scale=True`). Two conventions in one
table invites the question of whether the mixture is an accident worth removing.

It was measured rather than argued: the full 46-arm × 5-horizon grid was re-run with
the convention made uniformly covariance (`covpca_full`, 230 cells).

**The mixture is closer to the paper. Making it uniform is worse.**

| INDPRO, 120 deterministic cells | median \|Δ rel-RMSPE\| | beats AR,BIC |
|---|---|---|
| mixed (`far` covariance, `pca_step` correlation) | **0.0425** | 53 / 120 |
| uniform covariance | 0.0499 | 47 / 120 |
| paper | — | 67 / 120 |

8 cells moved closer, 17 moved farther. The damage concentrates exactly where it
should if correlation PCA is the right reading of the paper — on the arms that
consume `pca_step` factors:

| family | mixed | uniform | Δ |
|---|---|---|---|
| RFARDI | 0.0277 | 0.0490 | **−0.0213** |
| RRARDI | 0.0473 | 0.0646 | −0.0173 |
| Shrinkage (B1–B3) | 0.0478 | 0.0567 | −0.0089 |
| ARDI (`far`) | 0.0450 | 0.0450 | 0.0000 |

`ARDI` does not move, which is the arithmetic check: `far` was already covariance, so
the uniform run changes nothing for it.

**Reading.** GCLS standardize their predictor panel, so a correlation-PCA factor block
is the faithful reading for the arms that build factors from that panel. The mixture
is not an accident to be tidied away; it is two conventions each applied where the
paper's design calls for it. No change is made.

```{admonition} The control that makes this readable
:class: tip

The comparison has a built-in noise floor. 85 of the 225 cells belong to arms with
**no factor block at all** (`AR`, `RFAR`, `KRRAR`, `RRAR`, `SVR-AR`), which the PCA
convention cannot reach by construction. Splitting them:

```
deterministic (,POOS + AIC/BIC) : 0 of 60 moved,  max |delta| = 0.000e+00
,KF                             : 27 of 25 arms' cells moved, up to 4.8e-02
```

The deterministic half is **bit-identical across the two runs** — the strongest
available evidence that the pipeline changed only what the convention switch was
supposed to change. The `,KF` half moved anyway, on arms with no factors, which is
issue #513 measured a second time and from a different direction.

That is why the table above is restricted to the 120 deterministic cells. On all 225
the mixed convention still wins (0.0405 vs 0.0493), but 76 of the moves in that
figure are shuffle noise rather than convention.
```

---

```{admonition} What the re-run does NOT establish
:class: warning

It is tempting to read the movement from the stacked-panel table (INDPRO median
0.0421) to this one (0.0405) as the size of the `lags` fix. **That comparison is not
controlled and is not reported as one here.** The stacked-panel run predates version
control of `scripts/replication/gcls_2022_pipeline/` (committed 2026-08-07, `59da53b7`
— "None of this was under version control"), so the two runs may differ in more than
`lags`. A cell-by-cell diff shows the ambiguity directly: 68 of 225 cells moved closer
to the paper, 69 moved farther, and 88 did not move at all — including `AR,KF`, an arm
built with `predictors=[]` that the `lags` setting cannot reach at all.

The controlled evidence for the fix is the same-code A/B in the correction note above
— re-run on 2026-08-08 with `MF_GCLS_XLAGS` as the only difference, reproducing all six
figures and giving `max |delta| = 0` on the `predictors=[]` control arm — plus the direct
instrumentation of `MODEL_SPECS["far"].fit_func`, which showed the PCA input columns
literally as `['RPI_lag0', 'RPI_lag1', ...]`. That evidence is about the factor block, and
the corrected table reflects it where it applies: the ARDI family's median \|Δ\| is 0.0450
across its 20 cells, against 0.0882 under the stacked panel.

So the two claims are separated by the evidence available for each. **The factor-block
fix is controlled and holds.** The whole-table movement is not controlled and is not
claimed — and issue #513 now supplies a second reason it could not be, beyond the
untracked scripts: 105 of those 225 cells are CV-selected arms carrying run-to-run
noise larger than the effect.

The fix is justified because `X_t = Λ F_t + u_t` (eq. swardi2) is what the paper
writes — not because parity improved. Where parity did not improve, that is recorded
above rather than absorbed.
```

---

**1b — Accuracy by model family** (median |Δ rel-RMSPE| vs paper, full sample, all
five targets under the corrected `X_t`, all post-seed-fix).

| Model family | median \|Δ\| | n cells | previous (stacked panel) |
|---|---|---|---|
| RF (random forest) | **0.0202** | 100 | 0.0202 |
| CV-selected (POOS/KF) | 0.0278 | 100 | 0.0263 |
| Linear (AR/ARDI BIC/AIC) | 0.0371 | 75 | 0.0478 (over 100) |
| Shrinkage (Ridge/Lasso/EN) | 0.0479 | 550 | 0.0467 |
| KRR (kernel ridge) | 0.0683 | 100 | 0.0705 |
| SVR | 0.0746 | 200 | 0.0772 |
| **all 45 contenders** | **0.0479** | 1125 | 0.0473 |

Two things in this table are easy to miss.

**The overall number did not improve: 0.0473 → 0.0479.** The corrected `X_t` is
justified because eq. (swardi2) is what the paper writes, not because parity got
better; where it did not, that is recorded rather than absorbed. Families moved in both
directions — SVR, KRR and Linear closer to the paper, Shrinkage and CV-selected farther.

**Linear is 75 cells, not the 100 the superseded row claimed.** The Linear family has
four arms and one of them, `AR,BIC`, is the benchmark, so it is not a contender; the old
row counted it anyway. With the corrected denominator the six families partition the
1125 contender cells exactly (100 + 100 + 75 + 550 + 100 + 200), which the superseded
row set did not (it summed to 1150 against a stated total of 1125).

**1c — Accuracy by target** (median |Δ|, corrected `X_t`, post-seed-fix, 225 cells each):

| Target | median \|Δ\| | previous (stacked panel) |
|---|---|---|
| HOUST | **0.0341** | 0.032 |
| UNRATE | 0.0361 | 0.037 |
| INDPRO | 0.0389 | 0.042 |
| T10YFFM | 0.0679 | 0.065 |
| CPIAUCSL | 0.0768 | 0.077 |

The ordering is unchanged: HOUST and UNRATE reproduce best, CPIAUCSL worst. Only INDPRO
moved materially (0.042 → 0.0389), and §1a′ above explains why that movement is not
attributable to the `X_t` fix alone.

**Provenance of these five rows, because it decides whether they are comparable.**
INDPRO comes from the seeded re-run (`kfseed_accuracy.csv`, 2026-08-09), NOT from
`xlag0_full_accuracy.csv` (2026-08-07), which predates the #515 k-fold seed fix that
landed 2026-08-08 06:24. The other four targets come from `rest4_accuracy.csv` (920
cells, 2026-08-09 → 2026-08-11), also post-fix. Taking INDPRO from the older store would
have put one pre-seed target in a table with four post-seed ones, so the pre-seed figure
is reported separately instead:

| INDPRO, the same 225 cells | median \|Δ\| |
|---|---|
| unseeded (`xlag0_full`, 2026-08-07) | 0.0405 |
| seeded (`kfseed`, 2026-08-09) | **0.0389** |

Cell by cell, seeding moved 35 closer to the paper, 38 farther, and left 152 unchanged.
The median improved while more cells got worse — seeding removes run-to-run noise, it
does not move the design toward the paper. Reproducibility and accuracy are different
properties, and this is what that looks like in the numbers.

---

## Exhibit 2 — GCLS-2022 Table 2 (hyperparameter-selection / CV coefficients)

Treatment effect of the CV *selection method* on the target-variance-normalised
pseudo-R² (paper eq. r2_eq), estimated with the φ_{t,v,h} fixed effect absorbed by
within-transform + Driscoll–Kraay SE (same convention as the paper), over the 8
AR+ARDI arms. Baseline = BIC.

| Sub-panel | term | mine coef (DK se) | paper coef (se) |
|---|---|---|---|
| **All** | CV_KF | −2.545 (0.912)** | −0.038 (0.800) |
| | CV_POOS | −0.618 (1.534) | −1.351 (0.800) |
| | CV_AIC | −0.014 (0.770) | −0.509 (0.800) |
| **Data-rich (ARDI)** | CV_KF | −2.527 (1.378) | −0.314 (0.711) |
| | CV_POOS | −1.124 (2.768) | −1.440 (0.711) |
| **Data-poor (AR)** | CV_KF | −2.563 (1.006)* | 0.237 (0.411) |
| | CV_POOS | −0.112 (1.020) | −1.262 (0.411) |

**Reading.** The *sign/direction* reproduces (CV-based order selection tends to
underperform IC). The *magnitude* differs: my random K-fold **over-selects** at
long horizons (see below), so CV_KF is more negative than the paper's ≈0.
Diagnosis (store-level): at h=24, ARDI,K-fold selects n_lag=12 (grid max) where
BIC selects n_lag=1 → over-fit → pseudo-R² gap of −12.97 at ARDI/h24. This is the
known random-k-fold-on-time-series leakage (Bergmeir 2018, cited by the paper
§Feature 3). macroforecast's `random_kfold` is a *correct* standard k-fold and
also offers non-leaky temporal splitters (`explicit_folds`, `recursive_threefold`);
the registry uses `random_kfold` to stay faithful to the paper's "K-fold". The
exact magnitude is sensitive to the (under-specified) fold construction.

> **Note.** This Table-2 CV coefficient was degenerate (identically 0.000) until a
> package bug was fixed — see §Package bugs #5–#6. The −2.545 above is the
> post-fix, engaged value.

---

## Exhibit 3 — Treatment-effect figures (§"When are the ML Nonlinearities Important")

Per-feature treatment effects (controlled within+DK, pseudo-R²×100), full sample:

| Feature | mine coef | p | paper direction |
|---|---|---|---|
| X (big data / factors) | +8.05 | 0.021 * | positive, grows with horizon ✅ |
| NL (pooled RR\|RF\|KRR) | −0.77 | 0.73 | positive (NL matters) |
| — RF-only (data-rich) | **+11.02** | 0.019 * | ✅ **reproduces** |
| — KRR-only (data-rich) | **−7.74** | 0.020 * | ✗ paper: KRR≈RF, both help |
| SH (shrinkage) | −3.89 | 0.10 | ~neutral/negative |

**Reading.** RF-nonlinearity reproduces the paper's headline (nonlinearity helps,
data-rich). KRR-nonlinearity does **not** — my KRR under-performs the paper's on
the same factors, dragging the *pooled* NL to ≈0. This is a KRR-specific gap, fully
characterised in §Package verification. (SVR "loss-function" axis is dominated by
libsvm solver noise and is not a faithful comparison.)

---

## Exhibit 4 — GCLS-2022 Appendix B (Robustness of Treatment Effects)

**4a — By subsample and horizon** (treatment effects re-scored on the existing
forecasts, no re-run; USREC=recession, 56/455 months = 12%). This corresponds to
the paper's Appendix B/C subsample treatment-effect figures.

| Treatment effect | Full | **Recession** | Expansion | **Last-20yr (1998+)** | **Long-h (h≥9)** |
|---|---|---|---|---|---|
| X (big data) | +5.1 (ns) | **+31.9** (p=.0002) | +1.7 (ns) | +2.6 (ns) | +9.6 (p=.06) |
| NL (RR\|RF\|KRR) | −0.8 (ns) | +4.3 (p=.09) | −1.4 (ns) | +6.4 (p=.03) | −1.4 (ns) |
| RF-only NL (data-rich) | +11.0 * | +1.7 (noisy) | +12.2 * | **+22.5** (p=.002) | **+16.1** (p=.03) |

**Reproduces the paper's key robustness claims:** ML gains **concentrate in
recessions** (big-data X +31.9 in recessions vs +1.7 in expansions), and
**nonlinearity is strongest in recent data (RF-NL +22.5) and at long horizons
(RF-NL +16.1, X +9.6)** — matching the paper's finding that ML/big-data help most
in bad times, in the recent sample, and at longer horizons. (Pooled NL is dragged
by the KRR arm — see §Package verification; the reproducible RF-NL channel shows
the pattern cleanly.)

**4b — NN + Boosted-Trees robustness (Appendix D):** *see Exhibit 5 below.*

**4c — Rolling window (App B.2):** treatment-effect representatives (RR/RF/KRR ×
AR/ARDI) re-run under a **rolling** (not expanding) estimation window.

| Treatment effect | Expanding (baseline) | **Rolling (360-mo)** |
|---|---|---|
| X (big data) | +8.05 (p=.02) | **+12.03 (p=.003)** ✅ robust |
| RF-nonlinearity (data-rich) | +11.0 (p=.02) | +1.1 (ns) — attenuated |

The big-data effect X is **robust to the rolling window** (+12.0, significant),
reproducing the paper's App B.2 conclusion. The RF-nonlinearity effect is
attenuated under the shorter 360-month rolling estimation window (the tree
ensemble has less history to learn nonlinear structure); this is a property of
the rolling design, not a discrepancy in the effect's sign. *(A first attempt with a 240-month
window failed — see the em_factor finding below; a 360-month window runs cleanly.)*

> **Purpose-2 finding (new bug surfaced by this appendix).** `em_factor`
> preprocessing raises *"em_factor requires finite non-zero column standard
> deviations"* when a rolling estimation window makes some FRED-MD column
> constant (zero-std) inside the window — a case the expanding window (from 1960)
> never hits. This is adjacent to, but distinct from, the earlier all-NaN
> fit-column fix (`4bf4435c`): the column is finite but constant. **Proposed
> fix:** extend the fit-column prune to also drop zero-std columns. Workaround
> used here: a 360-month window (enough within-window variation).

**4d — Real-time vintages (App B.1):** **not attempted — data dependency.** The
GCLS-2022 replication runs on the single final-revised FRED-MD archive vintage
(2018-01). A faithful real-time exercise (paper window 2001M09–2017M12) requires
the *sequence* of monthly ALFRED / McCracken–Ng FRED-MD real-time vintages, which
is not currently staged on the host (analogous to the B3 futures-data acquisition
step). macroforecast *supports* real-time evaluation natively (`VintagePanelSpec`,
`custom_vintages`, day-one vintage-tagged caching — built and tested in B3); the
gap here is data acquisition, not package capability.

---

## Exhibit 5 — GCLS-2022 Appendix D (Nonlinearities Matter — NN + Boosted-Trees robustness)

The paper's robustness check that the nonlinearity result does not hinge on random
forests: it re-runs the exercise with a feed-forward neural net (NNARDI) and
boosted trees (BTARDI). We add both as native arms on the ARDI factor features and
measure the nonlinearity treatment effect = (arm pseudo-R²) − (linear ridge-ARDI
pseudo-R²), pooled over 5 targets × 5 horizons.

| Nonlinear ML method | NL treatment effect (mean) | median | reproduces? |
|---|---|---|---|
| Random forest (RFARDI) | **+11.13** | +3.98 | ✅ |
| Boosted trees (BTARDI) | **+8.87** | +1.99 | ✅ |
| Neural net (NNARDI) | **+6.24** | +2.49 | ✅ |

**All three nonlinear methods show a positive nonlinearity gain** over the linear
ARDI — reproducing the paper's Appendix-D conclusion that *"nonlinearities matter"*
is robust to the choice of ML method (not an RF artifact). This isolates the KRR
shortfall (§Package verification) as a KRR-specific replication-precision issue,
**not** a failure of the paper's headline nonlinearity claim, which reproduces
cleanly through RF, BT and NN. (torch installed into the worktree venv for the NN
arm; BT regularised to n_estimators=150, max_depth=4, lr=0.05, subsample=0.7 —
an unregularised depth-10 BT over-fits the ~660-obs windows.)

---

## Package verification — is the KRR gap a macroforecast defect? **No.**

The KRR family is the weakest-reproduced (Exhibit 1b, 0.071). We verified whether
this is a package fault:

1. **KRR math primitive is exact.** macroforecast's `kernel_ridge` is a thin
   `sklearn.kernel_ridge.KernelRidge` wrapper; predictions are **bit-identical** to
   raw sklearn (max |Δ| = 0.00e+00 across (γ,α) settings). sklearn's KRR is exactly
   the paper's eq. KT4 (raw y, RBF, no centering).
2. **Factors are correct.** The *linear* ridge on the same factors (RRARDI) matches
   the paper's ridge — so the factor construction is right for linear use.
3. **No constructible input reaches the paper's KRR number.** On UNRATE h12 (paper
   0.81, my CV-tuned 0.97) an *oracle* sweep (OOS-cheating best) over every
   constructible KRR input plateaus at **0.883**, never 0.81:
   σ/λ/n_f → 0.918; + input scaling → 0.934 (worse); + factor lags p_f (paper eq.
   KT2) → **0.885** (closes ~30% of the gap); + p_y + fine σ → 0.883 (plateau).

*(Configuration note: the KRR arms cross-validate **both** λ and σ jointly —
`krr_grid = {"alpha": …, "gamma": …}` — using the same native `SearchSpec`
param-grid mechanism the paper's SVR-RBF arms already use for their γ. This is
faithful to the paper's tuning set τ = {λ, σ, p_y, p_f, n_f}; it fixes an
otherwise unphysical σ-fixed blow-up on the level target T10YFFM at a small cost
in aggregate parity (0.061→0.071).)*

**Conclusion.** The ~0.07 residual is **not** reachable by any input macroforecast
can construct, and the KRR math + factors are verified correct. It is attributable
to the paper's exact, under-specified **factor-estimation pipeline** — to which the
RBF kernel is sensitive but linear ridge is invariant (which is why RRARDI matches
gold but KRRARDI does not, and why RF — robust to the factor count — fully
reproduces). This is a *replication-precision limit driven by paper under-
specification*, not a package bug.

---

## Package bugs found & fixed (Purpose 2)

Six defects were surfaced by this replication and fixed on `main` (byte-identical
golden gates on all behaviour-preserving changes):

1. **NW/HAC screening kernel** always crashed on a taper-name typo (`5c625368`).
2. **result_store** fail-safe: benchmark/arm rows silently dropped (`4df14838`).
3. **all-NaN fit column** emptied the fit sample on raw-wide predictors (`4bf4435c`).
4. **DM None-guard**: `float(None)` crash blanked native DM/MCS tables (`4f58d704`).
5. **CV-routing**: IC-owning models silently ignored an explicit CV SearchSpec
   (`376be08b`).
6. **CV degraded-guard**: a CV SearchSpec with its own splitter fell through to the
   degraded path (never ran CV) on windows with no validation block (`fadc1c20`).

**Operational gotcha (documented for the replication program):** after any code or
config change that alters forecasts, **both** the result_store cells **and** the
`_ckpt` checkpoint must be purged — otherwise a resume reconstructs stale
pre-change predictions (predictions-only, so params/model_spec come back `None`).

---

## Efficiency (Purpose 3/4)

All speedups are statistically identical (no draw/tolerance/tree reduction):
`n_jobs` cell-parallelism, `result_store` incremental resume, and the opt-in
K-prefix grouped evaluator for supervised-PCA. Throttling lesson: KRR/heavy runs
must cap `n_jobs` (an unthrottled `n_jobs=auto` OOM-killed the host once).

---

## Limitations / open [GAP]

- **KRR exact value** — was attributed to the paper's factor-estimation pipeline;
  see the 2026-08-07 correction above, which traces the factor residual to our own
  `lags` default instead. Pending re-measurement under the corrected panel.
- **CV-axis exact magnitude** — random-k-fold long-horizon over-selection, faithful
  to the paper's stated method; magnitude sensitive to under-specified folds.
- **far p_f=1** — the native `far` factor arm fixes factor-lag order at 1.
- **`em_factor` zero-std columns** — surfaced by the rolling appendix; proposed fix
  noted in Exhibit 4c (extend the fit-column prune to constant columns).
- **Appendix B.1 (real-time)** — not run: requires the ALFRED/McCracken real-time
  FRED-MD vintage sequence (data acquisition), not a package limitation.
- Density (PIT), absolute-loss, sign, quarterly (FRED-QD), Canadian appendices —
  out of current scope.

**Reproduced through Appendix D:** Table A1 (App A), Table 2, treatment-effect
figures, Appendix B (subsample robustness), Appendix C (additional subsample cuts),
Appendix D (NN + Boosted-Trees). Appendix B.2 (rolling) reproduced for the big-data
effect.

---

*Deliverable artifacts:* `runs/gcls_b4_stage1/{g2_indpro_accuracy.csv,
g2_rest_accuracy.csv}`, gold `gcls_tableA1_*_gold.csv`, comparison scripts in
`scripts/replication/gcls_2022_pipeline/` and this document. Store cells and
oracle stores are untracked (replication-workspace rule).
