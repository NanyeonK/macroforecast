# B4 (GCLS 2022) — parity diagnosis, 2026-08-04 night

Established from artifacts already on disk (no re-run needed). All numbers
re-derived tonight from `runs/gcls_b4_stage1/`.

## 1. The published absolute RMSPE row is on a different scale, by 12.2x

`published / ours` for the `AR,BIC (RMSPE)` row, all five targets x five horizons:

| target | h=1 | h=3 | h=9 | h=12 | h=24 |
|---|---|---|---|---|---|
| INDPRO | 12.04 | 11.66 | 12.75 | 12.37 | 12.39 |
| INF (CPIAUCSL) | 12.27 | 12.05 | 12.06 | 12.21 | 12.56 |
| HOUST | 11.99 | 12.21 | 12.48 | 12.29 | 12.51 |
| UNRATE | 12.14 | 11.94 | 12.38 | 12.36 | 11.96 |
| SPREAD (T10YFFM) | 11.98 | 11.91 | 11.86 | 12.25 | 12.75 |

**25 ratios: mean 12.215, median 12.212, sd 0.270 (2.2%), range 11.67-12.75.**

The five targets are of completely different natures — log growth rates
(INDPRO, INF, HOUST), a difference (UNRATE), and a percentage-point level
(SPREAD). **No modelling difference produces the same factor for a monthly log
growth rate and for a Treasury spread level.** This is a reporting-scale
convention.

`[GAP]` The paper does not state it. It is not annualization (that would not
apply to the level target), and it is not a percent conversion (that would be
100). It **cancels in the relative-RMSPE ratios**, which is what the parity table
compares, so it does not affect any parity conclusion — but anyone comparing the
absolute row directly would be misled.

## 2. The ratio gap decomposes into three layers

INDPRO, full sample, 225 cells (46 arms x 5 horizons, excluding the benchmark):

| layer | median abs delta | within 0.01 |
|---|---|---|
| plain AR family (`AR,POOS` .008, `AR,AIC` .014, `AR,KF` .019, `RRAR,KF` .009, `RFAR,POOS` .009) | **~0.01** | most |
| **factor-consuming arms** (ARDI, B1/B2/B3, `*-ARDI`) | **0.055** | 14/160 |
| SVR / KRR | 0.07-0.14 | few |

**The plain-AR layer at ~0.01 is the load-bearing evidence**: it says the data
build, the target construction, the AR,BIC benchmark and the evaluation are all
right. The gap is not diffuse — it is concentrated in the factor path and in the
two solver-based families (the latter an explicitly documented design risk).

Worst deterministic arms are all factor arms: `B1,ridge,POOS` .124,
`ARDI,POOS` .098, `B1,ridge,KF` .092, `ARDI,KF` .091, `ARDI,BIC` .086.

## 3. The factor-path gap grows with the horizon

`ARDI,BIC` / `ARDI,AIC`, INDPRO, full sample:

| h | published | ours (standardized PCA) | abs delta |
|---|---|---|---|
| 1 | 0.946 / 0.959 | 1.015 / 0.976 | .069 / .017 |
| 3 | 0.991 / 0.968 | 1.004 / 0.993 | **.013 / .025** |
| 9 | 1.037 / 1.017 | 1.142 / 1.122 | .105 |
| 12 | 1.004 / 0.998 | 1.090 / 1.083 | .085 |
| 24 | 0.968 / 0.943 | 1.090 / 1.085 | .122 / .142 |

Both agree ARDI does not beat AR at long horizons; ours says it more strongly.
A horizon-growing gap is **not** the signature of a PCA-convention difference,
which would shift things roughly uniformly. It is the signature the GCLS-2021
replication already documented once (direct-policy stale persistence, fixed in
PRs #413-#422) — worth re-checking against that precedent.

## 4. Why the convention question cannot be settled from the sources

- The paper's own TeX was searched end to end: it specifies direct forecasting
  and the h-step target exactly (`(1/h)ln(Y_{t+h}/Y_t)` for I(1), level for
  I(0)) — **both of which our runner matches** — but never states whether X is
  standardized before the PCA.
- The author archive (`jae_2910_glss_replication_20260602`) is **data only**:
  268 files, zero code. Its readme confirms the transforms are left to the
  reader.

So the convention is an irreducible `[GAP]` from the published material. Per the
standing rule this is handled by producing **both** variants and reporting them
side by side, never by choosing the one that matches.

## 5. In flight

`MF_GCLS_PCA_SCALE=0` (covariance PCA) on `ARDI,BIC` + `ARDI,AIC` + `AR,BIC`,
INDPRO, 5 horizons, 6 workers. Default (`scale=True`) unchanged.

## Runner issues found

- `--arms` split on `","` but arm names contain commas (`ARDI,BIC`), so the flag
  could not express them. `;` is now accepted as a separator.
- `--out-prefix` is joined onto `OUT_DIR`, so passing a path nests it.

## 6. Hypotheses raised and ELIMINATED tonight

Recorded so they are not re-litigated. Both were mine, and both are dead.

**(a) The PCA standardization convention explains the factor gap.** Raised
because `pca_step(scale=True)` is the package default, the registry never
overrides it, and B3 (ZWW) found a paper in this literature whose headline turned
on exactly that choice. **Weakened by the shape of the gap**: it grows with the
horizon (h=3: 0.013-0.025; h=24: 0.122-0.142), and a convention difference in the
factor extraction would shift things roughly uniformly, not with h. The variant
run (`MF_GCLS_PCA_SCALE=0`) will settle it with data rather than argument -- the
switch is worth keeping either way, because the convention is genuinely
unstated.

**(b) The GCLS-2021 direct-policy stale-persistence bug has recurred.** Raised
because that replication documented the *same* signature -- "this corrupted every
direct relative-RMSE and grew with the horizon" (`gcls_2021_replication.md:100`)
-- and because B4's INDPRO target uses `direct_average`, while the fix's own
description names "the direct policy". **Eliminated by checking the wiring rather
than the prose**: `_AR`'s docstring says `direct=True` applies to "the
direct/direct_average policies", `direct_capable` is computed as
`"direct" in model_spec.default_params`, and `far` -- which is what the ARDI arms
resolve to (registry:46) -- declares `default_params = ['n_factors', 'n_lag',
'random_state', 'direct']`. So the direct-projection mode IS active for these
arms. The 2021 bug is not back.

**What is NOT yet explained**: why the factor-path gap grows with the horizon.
The plain-AR layer matching to ~0.01 rules out the data, the target
construction, the benchmark and the evaluation. The two eliminations above rule
out the two most obvious factor-path causes. This is the open question B4 should
be picked up on.

## 7. The convention question is SETTLED, and it exposed a package gap

The variant run finished. **Both conventions produced identical numbers to three
decimals in all ten cells.** That is not "the convention does not matter" -- it is
the switch never reaching ARDI, and finding out why answered the question:

| route | factor-extraction convention | exposed? |
|---|---|---|
| `far` -- **what the ARDI arms resolve to** (registry:46, "PCA inside the model") | **covariance** (`(block - block.mean()).fillna(0)`, no division by std; `timeseries.py:1764-1767`) | **no** -- `far(X, y, *, n_factors, n_lag, random_state, direct)` has no scaling parameter at all |
| `pca_step` -- what B1/B2/B3 and SVR-ARDI use | **standardized/correlation** (`scale=True` default) | yes |

So **ARDI was already on the covariance convention**, and its gap is 0.086 median
regardless. `MF_GCLS_PCA_SCALE` only ever touched the `pca_step` route.

**Hypothesis (a) is therefore settled as NEGATIVE for ARDI**: the factor-extraction
convention does not explain the horizon-growing gap. The switch is still worth
keeping for the `pca_step` arms (B1/B2/B3), where the convention is genuinely
unstated by the paper and genuinely selectable.

### Package gap worth filing

**The package uses two different factor-extraction conventions depending on the
route, and one of them cannot be changed.** A user who writes `far` gets
covariance PCA; a user who writes `pca_step` gets correlation PCA; nothing warns
that these differ, and `far` offers no way to ask for the other. B3 (ZWW)
established that this exact choice can decide a paper's headline -- so a
replication that needs the standardized convention inside a factor-augmented
autoregression currently cannot express it.

### Cost note

The first variant run computed for ~95 minutes and then died writing its output,
because `--out-prefix` is joined onto `OUT_DIR` and I passed a path -- a failure I
had predicted and not fixed before launching. **The ResultStore saved it**: the
relaunch with the correct prefix reproduced every cell from cache in 90 seconds.
That is the incremental-store design doing exactly what it is for.

## 8. The p_f deviation, and what is actually new about it

**This deviation was already known.** `registry.py:23-40` documents it at the top
of the file: "``far`` builds CONTEMPORANEOUS PCA factors ...; it does not expose a
separate factor-lag order. GCLS's p_f in {1,3,6,12} grid is therefore NOT native."
It was an accepted, recorded bound on the replication, not an oversight.

Three things here are new:

1. **It is measured, not assumed** -- see below.
2. **It is the leading candidate for the horizon-growing gap.** The prior note
   recorded the deviation but did not connect it to the parity shape.
3. **The composable route works today.** The registry note proposes a package
   extension ("model_selection that can search FEATURE-construction params" OR
   "an ARDI model wrapper"), but `pca_step` + `lag_step` already builds the
   paper's design; what is lost is only the IC-searchability of `n_f`, not the
   design itself.


The paper defines ARDI explicitly (eq. `swardi1`, TeX:276):

```
y_{t+h} = c + rho(L) y_t + beta(L) F_t + e_{t+h}
```

"where `rho(L)` and `beta(L)` are lag polynomials of orders **p_y** and **p_f**",
and TeX:364 lists the tuning set as `tau = {lambda, sigma, p_y, p_f, n_f}`. So
the factor block enters with its own **tuned lag order**.

**Our ARDI has p_f pinned at 0.** Measured, not inferred: `far(X, y,
n_factors=3, n_lag=2, direct=True)` produces exactly **5** coefficients --
3 contemporaneous factors + 2 own lags. There are no `F_{t-1}`, `F_{t-2}`
regressors, and `far` exposes no parameter that would add them.

The B4 registry (registry:46) routes ARDI to native `far` precisely because the
PCA is "inside the model". That choice silently drops one of the paper's three
tuned dimensions.

### This is our spec error, not a package limitation

The package CAN express the paper's ARDI, through the composable route:

```python
feature_spec(
    target="y", target_lags=p_y,
    steps=[
        pca_step(name="fac", input="panel", n_components=n_f,
                 include=False, scale=False),      # far's own convention
        lag_step(name="faclag", input="fac", lags=range(p_f + 1), include=True),
    ],
)
# -> fac1_lag0..fac3_lag2, y_lag1, y_lag2   == beta(L)F_t + rho(L)y_t
```

Verified to build exactly those columns. `scale=False` matches what `far` does
internally (issue #495), so switching route does not silently also switch the
factor-extraction convention.

### Why this is the leading candidate for the horizon-growing gap

The gap is concentrated in factor arms and grows with h (h=3: 0.013-0.025;
h=24: 0.122-0.142). Omitted factor lags fit that shape: the longer the horizon,
the more of the factors' predictive content sits in their lags rather than their
current value, so pinning `p_f = 0` costs more as h grows. The two hypotheses
already eliminated (PCA convention, stale persistence) do not have that shape.

**Not yet proven** -- the next step is to run the corrected ARDI at INDPRO x 5
horizons and compare against both the published table and the current arm.

### Result: p_f is NOT the explanation -- hypothesis eliminated

Controlled run, `ARDI,BIC`, INDPRO, full sample. The middle column is the
control: same route, same fixed `n_f`, so the only difference from the third is
`p_f`.

| h | published | far (IC n_f, p_f=0) | comp (n_f=6, p_f=0) | comp (n_f=6, p_f=2) |
|---|---|---|---|---|
| 1 | 0.946 | 1.015 | 0.990 | 0.997 |
| 3 | 0.991 | 1.004 | 0.990 | 1.010 |
| 9 | 1.037 | 1.142 | 1.125 | 1.144 |
| 12 | 1.004 | 1.090 | 1.072 | 1.108 |
| 24 | 0.968 | 1.090 | 1.041 | 1.114 |

median abs deviation: far **0.0856**, ctrl **0.0682**, p_f=2 **0.1040**.

**Adding the paper's factor lags moved AWAY from the published value at every
single horizon (0/5 closer), and by more at long horizons** (h>=9 median .0728 ->
.1065). So the omitted `p_f` does not explain the horizon-growing gap; it was a
plausible shape-match and it is wrong. Fourth hypothesis eliminated.

The gap still grows with h in all three variants, so that structure is robust to
this change.

### Side finding worth following: IC-selected n_f is WORSE than a fixed n_f

The control differs from the baseline in exactly one other way -- it fixes
`n_f = 6` where `far` IC-selects it from (3, 6, 10) -- and it is **closer to the
published table at every horizon** (median .0682 vs .0856, and .0728 vs .1221 at
h=24).

That points the next investigation at the hyperparameter SELECTION rather than
at the model form: if our per-origin IC is choosing a factor count the paper's
procedure would not, that would show up as exactly this kind of drift, and it
would plausibly worsen with the horizon as the effective sample for the h-ahead
target shrinks.

## 9. The paper states its ARDI grid, and our p_f is not in it

Appendix, TeX:1767, verbatim:

> `p_y \in \{1,3,6,12\}`, `K \in \{3,6,10 \}`, `p_f \in \{1,3,6,12\}`

| parameter | paper | ours |
|---|---|---|
| `p_y` | {1, 3, 6, 12} | (1, 3, 6, 12) -- **exact match** |
| `K` (= n_f) | {3, 6, 10} | (3, 6, 10) -- **exact match** |
| `p_f` | **{1, 3, 6, 12}** | **pinned at 0** |

Two of the three grids were already right. The third is not merely unsearched --
**our value is outside the paper's grid entirely**. In the paper's notation the
factor block is `{X_{t-j}}_{j=0}^{p_f}` (eq. Hs), so their smallest setting,
`p_f = 1`, already carries `F_t` AND `F_{t-1}`; native `far` carries `F_t` alone,
which is `p_f = 0`.

**That also invalidates the p_f test in section 8 as a test of the paper's
spec.** It used `p_f = 2`, which is not in {1, 3, 6, 12} either. Its conclusion
stands as far as it goes -- adding two factor lags moved away from the published
value at every horizon -- but it did not test a setting the paper would ever
select. Re-running at `p_f = 1` (their minimum, and the nearest grid point to our
current 0) and `p_f = 3`.

The paper also selects `p_f` per origin from that grid rather than fixing it, and
the composable route cannot search it (registry:23-40 records why: `SearchSpec`
searches MODEL params, and `p_f` is a feature-construction param). So even a
correct fixed `p_f` is a bounded approximation of the paper's procedure.

## 10. ROOT CAUSE CANDIDATE: the factor panel is stacked, not a cross-section

Measured by instrumenting `MODEL_SPECS["far"].fit_func` on the real INDPRO cell:

```
raw FRED-MD predictor series passed to the arm : 128
columns in the design matrix handed to far     : 277
columns far actually PCAs                      : 264
first PCA-input columns: ['RPI_lag0', 'RPI_lag1', 'W875RX1_lag0', 'W875RX1_lag1', ...]
```

**264 = 132 series x {t, t-1}.** The paper's factor model is `X_t = Lambda F_t + u_t`
(eq. swardi2) -- factors of the cross-section at ONE time index. Ours are the
principal components of a STACKED `[X_t, X_{t-1}]` panel, which is a different
object.

### Cause

`registry.py::_far_features` calls `mf.feature_spec(target=..., predictors=cols,
target_lags=TARGET_LAGS)` without `lags`, and **`feature_spec`'s default is
`lags=(0, 1)`**. So every predictor entered at t and t-1.

### What it explains

- **Why adding `p_f` made things worse** (section 8): the lagged information was
  ALREADY inside the factors. Explicit factor lags double-counted it.
- **Why the gap grows with the horizon**: a stacked panel needs more components
  to span the same information, so the IC reaches for larger K -- and it reaches
  furthest at long h, where the h-period average target is smoothest and extra
  components always buy in-sample fit. That matches the measured selection drift
  (n_factors=10 in 78% of origins at h=24, 0% at h=1).
- **Why it is confined to factor arms**: AR arms never touch the panel.

### Is this a package defect?

**No -- it is our replication spec.** `feature_spec(lags=(0, 1))` is a defensible
general default; the registry simply never stated the paper's spec. The fix is
one keyword.

There is a narrower package-side question worth separating: for a model that runs
PCA INTERNALLY on whatever design it is handed (`far`, and anything like it),
silently receiving a lag-stacked design changes what its factors MEAN, with no
diagnostic. Whether that deserves a warning is a judgement to make after the
corrected run, not before.

### Verification in flight

`MF_GCLS_XLAGS=0` (the paper's `X_t`) vs the stacked baseline, INDPRO x 5
horizons, against the published Table A1.

### CONFIRMED: the stacked factor panel WAS the horizon-growing gap

`ARDI,BIC`, INDPRO, full sample, against published Table A1:

| h | published | stacked `[X_t, X_{t-1}]` | paper `X_t` only |
|---|---|---|---|
| 1 | 0.946 | 1.015 | 0.990 |
| 3 | 0.991 | 1.004 | 1.010 |
| 9 | 1.037 | 1.142 | 1.100 |
| 12 | 1.004 | 1.090 | 1.050 |
| **24** | **0.968** | **1.090** | **0.986** |

| | stacked | paper `X_t` |
|---|---|---|
| median abs deviation | 0.0856 | **0.0441** |
| h >= 9 | 0.1051 | **0.0459** |
| worst cell | 0.1221 | **0.0627** |
| closer to published | -- | **4 of 5 horizons** |

**The horizon-growing signature is gone.** Stacked deviates 0.069 -> 0.105 ->
0.086 -> 0.122 as h grows; the paper's spec goes 0.044 -> 0.063 -> 0.046 ->
**0.018**. At h=24 the deviation falls by a factor of seven.

That is the signature this whole investigation was chasing, and it was one
missing keyword: `feature_spec(..., lags=0)`.

### Accounting: what was actually wrong

Nothing in the package. Six hypotheses were tested and eliminated -- data,
target construction, benchmark, evaluation, PCA convention, stale persistence,
BIC formula, parameter count, fit-sample alignment, and the selection procedure
all check out against the paper. **Both real deviations were in the replication
spec**, in the same place: the registry did not state the paper's factor design.

1. `lags` left at the package default `(0, 1)` -> the PCA ran on a stacked panel.
   **This one is the cause.**
2. `p_f` pinned at 0, outside the paper's grid {1, 3, 6, 12}. Still a deviation,
   still to be corrected -- but note section 8 tested it ON TOP of the stacked
   panel, where the lag information was already double-counted. **That test must
   be re-run now that the panel is right**; its negative result cannot be
   trusted as a statement about the paper's spec.

### Provenance warning

`registry.py` now defaults to `X_LAGS = (0,)`. Every artifact produced BEFORE
this change used the stacked panel and is not comparable to anything produced
after it.
