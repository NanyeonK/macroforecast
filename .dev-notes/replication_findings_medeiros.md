# Medeiros (2021) Replication Findings

## 2026-07-08 -- B1 G2 full pipeline after oracle correction

### Parity Status

Acceptance oracle is IJF Table 5, not stale `qa/g2_rw_ar_v2.out`. The runner now enters
`pipeline_spec(..., n_jobs="auto", result_store="qa/result_cells",
preprocessing_cache_dir="qa/prep_cache", seed=42, save_models=False)` and preserves the
paper rolling formula with a runner-local `WindowSpec` subclass:

| Regime | Test dates | Base | R(h=1) | R(h=3) | R(h=6) | R(h=12) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| s1 | 1990-01..2000-12 | 360 | 354 | 352 | 349 | 343 |
| s2 | 2001-01..2015-12 | 492 | 486 | 484 | 481 | 475 |

Full four-arm run output, scored against paper Table 5:

| Arm | h=1 ratio / paper / d | h=3 ratio / paper / d | h=6 ratio / paper / d | h=12 ratio / paper / d |
| --- | ---: | ---: | ---: | ---: |
| AR | 0.911304 / 0.902 / +0.009304 MATCH | 0.789546 / 0.790 / -0.000454 MATCH | 0.792246 / 0.791 / +0.001246 MATCH | 0.764836 / 0.753 / +0.011836 CLOSE |
| UCSV | 0.911726 / 0.954 / -0.042274 CLOSE | 0.726410 / 0.797 / -0.070590 DIVERGENT | 0.720370 / 0.777 / -0.056630 DIVERGENT | 0.695099 / 0.781 / -0.085901 DIVERGENT |
| RF | 0.915638 / 0.844 / +0.071638 DIVERGENT | 0.741989 / 0.706 / +0.035989 CLOSE | 0.736565 / 0.715 / +0.021565 CLOSE | 0.750681 / 0.685 / +0.065681 DIVERGENT |

RW baseline diagnostics from the same score table:

| h | OOS n | RW RMSE |
| ---: | ---: | ---: |
| 1 | 311 | 0.2889 |
| 3 | 309 | 0.3847 |
| 6 | 306 | 0.3852 |
| 12 | 300 | 0.3990 |

P4 effect: AR remains stable at `0.911304 / 0.789546 / 0.792246 / 0.764836` against the
paper oracle and the prior low-level diagnostic already showed pipeline == current
low-level semantics. RF/UCSV were run; no CSR/JMA/LASSO/bagging/hybrids/G3 were run.

### Efficiency

Before, measured/extrapolated sequential single-arm loop:

| Arm | Sequential basis | Seconds |
| --- | --- | ---: |
| RW | measured 4 horizons | 310.0 |
| AR | measured 4 horizons | 379.2 |
| RF | measured 4 horizons | 6465.7 |
| UCSV | extrapolated 312 origins x 4h x 6.99s/origin | 8700.0 |
| Total | measured/extrapolated G2 | 15854.9 sec = 4.40 h |

After, successful pipeline scoring command with `qa/result_cells` resume:

| Run segment | Seconds |
| --- | ---: |
| run_pipeline s1 | 2.7 |
| run_pipeline s2 | 2716.8 |
| Total scoring command | 2719.5 |
| Validated speedup vs sequential | 5.83x |

Cold/resume caveat: the first full command completed cold `s1` in `1504.7` seconds, then
idled in `ProcessPoolExecutor.map` during `s2` after writing partial result-store cells. It
was interrupted and resumed with the same full command. Completed-cell wall-clock lower
bound is `1504.7 + 2716.8 = 4221.5` seconds (`70.4` minutes), a conservative `3.76x`
speedup over the 4.40h sequential estimate.

### Bugs / Gaps

- `[PARITY, high]` RF h=1/h=12 and UCSV h=3/h=6/h=12 are outside the requested paper-T5
  tolerance. AR P4 passes, but RF/UCSV need protocol diagnosis before paper claims.
- `[EXECUTOR, medium]` first full command idled in `ProcessPoolExecutor.map` during `s2`
  after partial cell writes. Result-store resume completed successfully, so this is a
  runner/pipeline reliability gap rather than a parity-number drift.
- `[SILENT-WRONG RISK, medium]` `macroforecast/models/tree.py:66`: registered
  `random_forest` defaults to `n_estimators=200`, `max_features=None` (sklearn all
  features), `random_state=0`. The Medeiros author-code/R `randomForest` convention is
  `ntree=500`, `mtry=floor(p/3)`. Replication runner passes explicit
  `{"n_estimators": 500, "max_features": 1.0/3.0, "random_state": 42}` and omits
  `n_jobs`.
- `[EFFICIENCY, high]` the old model x horizon low-level loop left cell-level parallelism
  unused and recomputed shared work. The pipeline path validates material speedup even with
  the conservative cold/resume caveat.
- `[WINDOW/API GAP, medium]` one `PipelineSpec.window` cannot express Medeiros's
  horizon-specific rolling estimation size directly. The runner uses a module-level,
  pickleable `WindowSpec` subclass that delegates to `from_cutoffs()` after the pipeline
  injects the cell horizon.
- `[POLICY/API GAP, low]` the already-transformed parquet panel has no t-code metadata, so
  `TargetSpec("CPIAUCSL")` fails. The runner must use
  `TargetSpec("CPIAUCSL", transform="level", policy="direct")`.
- `[POLICY GUARD, low]` `naive` and `ucsv` are direct-policy guarded models. To reproduce
  the validated direct RW/UCSV replication protocol, the runner sets
  `on_unsupported_direct="warn"`.
- `[RESULT-STORE GAP AVOIDED]` RF features include a custom November 2008 dummy. The custom
  step has `__mf_digest__ = "medeiros-nov2008-dummy-v1"` so RF cells are digestible in
  `qa/result_cells`.

## 2026-07-09 -- B1 G2e RF/UCSV divergence diagnosis

### Post-Fix Parity Status

Active cached `run_pipeline(..., n_jobs="auto", result_store="qa/result_cells")` score after
the RF author-spec fix:

| Arm | h=1 ratio / paper / d | h=3 ratio / paper / d | h=6 ratio / paper / d | h=12 ratio / paper / d |
| --- | ---: | ---: | ---: | ---: |
| AR | 0.911304 / 0.902 / +0.009304 MATCH | 0.789546 / 0.790 / -0.000454 MATCH | 0.792246 / 0.791 / +0.001246 MATCH | 0.764836 / 0.753 / +0.011836 CLOSE |
| UCSV | 0.911726 / 0.954 / -0.042274 CLOSE | 0.726410 / 0.797 / -0.070590 DIVERGENT | 0.720370 / 0.777 / -0.056630 DIVERGENT | 0.695099 / 0.781 / -0.085901 DIVERGENT |
| RF | 0.907273 / 0.844 / +0.063273 DIVERGENT | 0.740947 / 0.706 / +0.034947 CLOSE | 0.730065 / 0.715 / +0.015065 CLOSE | 0.737382 / 0.685 / +0.052382 DIVERGENT |

RW baseline remained `0.2889 / 0.3847 / 0.3852 / 0.3990` with OOS n
`311 / 309 / 306 / 300`.

### RF Author-Spec Diagnosis

Author source:

- `qa/ForecastingInflation/03_call_model.R:14-16` sets Table-5 `model_name = "RF"` and
  `model_function = runrf`.
- `qa/ForecastingInflation/functions/functions.R:114-127` defines `runrf()` and calls
  `randomForest::randomForest(Xin, yin, importance = TRUE)`.
- `qa/ForecastingInflation/functions/functions.R:130-160` defines `runrfols()`; its
  `maxnodes = 25, ntree = 500` call is for RF/OLS only, not the plain RF Table-5 row.

Exact plain-RF author controls are therefore the R `randomForest` regression defaults:
`ntree=500`, `mtry=floor(p/3)`, `nodesize=5`, `replace=TRUE`, `sampsize=nrow(Xin)`, and
`maxnodes=NULL`. The package wrapper can express these via
`macroforecast/models/tree.py:62-100`, including sklearn pass-through kwargs. The runner now
sets:

```python
{
    "n_estimators": 500,
    "max_features": 1.0 / 3.0,
    "min_samples_leaf": 5,
    "bootstrap": True,
    "max_samples": None,
    "max_leaf_nodes": None,
    "random_state": 42,
}
```

and explicitly disables default RF model selection with
`model_selection={"random_forest": None}`.

Effect: the author-spec RF rerun improved the divergent RF cells but did not fully close
them:

| Cell | Before | After | Paper | Status |
| --- | ---: | ---: | ---: | --- |
| RF h=1 | 0.915638 | 0.907273 | 0.844 | DIVERGENT |
| RF h=12 | 0.750681 | 0.737382 | 0.685 | DIVERGENT |

Classification: the missing runner controls were a config diff and are now corrected. The
remaining RF divergence is a package/protocol gap, not an omitted `maxnodes=25` knob.
Likely contributors are the sklearn-vs-R `randomForest` backend and the already noted
feature-construction delta in `scripts/replication/medeiros_2021_pipeline/registry.py`
where macroforecast lags raw predictors and adds PCA factors at lag 0, while the author
embeds series plus factors before fitting.

### Package Bugs / Gaps

- `[REPLICATION-FIDELITY BUG, high]`
  `macroforecast/forecasting/selection_stage.py:349-361`,
  `macroforecast/forecasting/policies/base.py:128-133`, and
  `macroforecast/model_selection/search.py:95-107`: `model_selection=None` silently activates
  model-owned default search when a registered model has `search_spaces`. A replication arm
  with fixed author `Arm.params` is therefore not fixed unless the caller passes a mapping
  such as `model_selection={"random_forest": None}`. Repro from this run: after adding
  author RF tree controls but before disabling model selection, the RF ratios were unchanged
  from the earlier tuned run and decoded result-store `params` varied by origin
  (`n_estimators` 200/500, `max_depth` 3/5/None, `min_samples_leaf` 1/3/5). Fix lane:
  provide an explicit fixed-params mode or warn/error when default search will override
  caller-supplied paper parameters.
- `[RF BACKEND/FEATURE PROTOCOL GAP, high]` after author controls and no-search execution,
  RF h=1/h=12 remain outside paper tolerance. No missing wrapper knob was found: sklearn
  kwargs carry `max_leaf_nodes`, `bootstrap`, and `max_samples`. The remaining issue should
  be resolved by an R-compatible randomForest backend or by eliminating the PCA/embed-order
  feature delta before promoting RF as a faithful Table-5 replication.
- `[UCSV PROTOCOL GAP, high]` paper Appendix B.1 (`qa/medeiros_correct_fulltext.txt` around
  lines 1843-1852 after null stripping) states the UCSV equations, inverse-gamma variance
  priors, `Vtau=Vh=0.12`, MCMC estimation, one-sided `tau_t|t` h-step forecasts, and
  twelve-month inflation for accumulated forecasts. The author repos contain no UCSV code
  (`rg "UCSV|ucsv|Stock|Watson|gamma|MCMC|Vtau|Vh|0\\.12" qa/ForecastingInflation
  qa/HDeconometrics` returned no hits), and neither source publishes draw count, burn-in, or
  a package-compatible log-volatility `gamma`. Macroforecast exposes only
  `n_draws`, `burn`, `gamma`, and `random_state` in `macroforecast/models/bayesian.py:255-318`;
  it cannot express the paper's separate initial-prior variances as written.
- `[UCSV LOOKAHEAD CHECK, no new bug found]`
  `macroforecast/forecasting/selection_stage.py:35-67` filters direct-policy training targets
  to labels with positions `<= origin_pos - horizon`, and
  `macroforecast/models/bayesian.py:61-115` fits only the supplied target sample and predicts
  the posterior mean final trend. No lookahead/centering path was found for the UCSV cells.

### Efficiency

- Result-store reuse worked as intended. The final all-G2 cached score took `2.7` seconds
  for s1 and `4.1` seconds for s2 (`6.8` total) and reused RW/AR/UCSV plus existing RF cells.
- The intermediate RF rerun with default package RF search still enabled took `679.4`
  seconds for s1 and `931.8` seconds for s2 (`1611.3` total) and was statistically wrong for
  this paper because it tuned over default search-space values.
- The final fixed-author RF-only rerun took `130.2` seconds for s1 and `181.9` seconds for
  s2 (`312.2` total). Disabling default search was not a shortcut: it restored the author's
  fixed RF specification while reducing wall-clock by about `5.16x` relative to the
  accidentally tuned RF rerun.

### P4 / Statistical Identity Guard

No speed shortcut was used: RF still used 500 trees, UCSV draw/burn counts were not reduced,
and no tolerance loosening, subsampling, or coarser grid was introduced. AR/RW result-store
numbers stayed unchanged, and no `macroforecast/**` package code was edited.

## 2026-07-09 -- B1 G2f RF feature-matrix protocol fix

### Verdict

The residual RF gap was a **PROTOCOL** difference, not an unavoidable sklearn-vs-R backend
caveat. The runner did not previously build the exact `dataprep()` matrix fed to R
`randomForest`.

Post-fix RF row against IJF Table 5:

| Arm | h=1 ratio / paper / d | h=3 ratio / paper / d | h=6 ratio / paper / d | h=12 ratio / paper / d |
| --- | ---: | ---: | ---: | ---: |
| RF | 0.859709 / 0.844 / +0.015709 CLOSE | 0.732985 / 0.706 / +0.026985 CLOSE | 0.749831 / 0.715 / +0.034831 CLOSE | 0.710726 / 0.685 / +0.025726 CLOSE |

The previously divergent cells moved from `0.907273` to `0.859709` at h=1 and from
`0.737382` to `0.710726` at h=12. RF is now within the RF acceptance tolerance at all four
horizons.

### Author Matrix Spec

Source evidence:

- `qa/ForecastingInflation/03_call_model.R:14-16` sets `model_name = "RF"` and
  `model_function = runrf`.
- `qa/ForecastingInflation/functions/functions.R:114-127` calls
  `randomForest::randomForest(Xin, yin, importance = TRUE)`.
- `qa/ForecastingInflation/functions/functions.R:4-53` defines `dataprep()`.

For target `CPIAUCSL` and direct horizon `h`, the author matrix is:

- `df = df[ind,]`, where `df` includes `CPIAUCSL`.
- `factors = princomp(scale(df))$scores[,1:4]`; PCA is fitted on the full in-window panel,
  including the target. R `scale()` uses sample standard deviations and `princomp()`
  applies `fix_sign=TRUE`.
- `x = cbind(df, factors)` for plain RF.
- `X = embed(as.matrix(x), 4)`, so the order is all series and factors at lag 0, then all
  series and factors at lag 1, then lag 2, then lag 3.
- `Xin = X[-((nrow(X)-h+1):nrow(X)),]`; `Xout = X[nrow(X),]`;
  `yin = tail(df[, "CPIAUCSL"], nrow(Xin))`.
- The Nov-2008 dummy is appended to `Xin` if the date is inside `yin`; `Xout` receives a
  final zero dummy.

Author data selection is defined in `qa/ForecastingInflation/01_get_fred_data.R:12-36`:
FRED-MD `current.csv`, 1960-01 onward, transformed price tcode-6 series, then
`select_if(~ !any(is.na(.)))`. The author clone does not include `data/data.rda`; the local
replication panel `qa/medeiros_panel.parquet` has 112 balanced transformed columns.

### Runner Diff

Previous `base_features()` in `scripts/replication/medeiros_2021_pipeline/registry.py`
built:

- raw predictor lags `0..3`;
- target lags `0..3` as a separate block;
- a PCA step at lag 0 only;
- a custom Nov-2008 dummy.

Differences from the author matrix:

- PCA omitted `CPIAUCSL` because macroforecast's normal predictor resolution excludes
  target columns.
- PCA factors were not lagged at `1..3`.
- Column order was macroforecast per-series lag order, not R `embed` lag-block order.
- The previous PCA implementation used the package PCA step rather than R
  `scale()` + `princomp(fix_sign=TRUE)` semantics.

No row/window divergence was found in this pass: the runner retains the existing
horizon-specific rolling sizes `s1: 354/352/349/343` and `s2: 486/484/481/475` for
h=`1/3/6/12`, and the RF-only rerun produced the same RW RMSE and OOS counts as before.

### Fix Applied

Only the replication runner was patched:

- `scripts/replication/medeiros_2021_pipeline/registry.py:51-83` implements R-style
  `princomp(scale(df))` scores.
- `scripts/replication/medeiros_2021_pipeline/registry.py:86-123` fits/transforms the
  author RF matrix from the local panel, including target-in-PCA, factor lags, R `embed`
  lag-block order, and the Nov-2008 dummy.
- `scripts/replication/medeiros_2021_pipeline/registry.py:126-145` makes the custom feature
  step digestible and uses it in `base_features()`.

No `macroforecast/**` package file was edited. No blocking feature-API gap remains for this
replication because the existing custom feature-step API could express the author matrix.
The standard shortcut/step API would not have matched the matrix by itself, but that did not
block a runner-local author-faithful configuration.

### Rerun / P4 Guard

Command run:

```bash
python3 scripts/replication/medeiros_2021_pipeline/run_block.py --arms rf
```

The runner includes `rw` automatically for scoring; `qa/result_cells` reused RW cells and
recomputed the changed RF feature-identity cells only. No UCSV, AR, G3, CSR/JMA/LASSO,
bagging, or hybrid arms were run.

Runtime:

| Segment | Seconds |
| --- | ---: |
| pipeline s1 | 228.0 |
| pipeline s2 | 417.0 |
| pipeline total | 645.0 |

P4 guard: RF still used 500 trees, `mtry=p/3`, `nodesize=5`, `replace=TRUE`,
`maxnodes=NULL`, fixed seed 42, and `model_selection={"random_forest": None}`. No
tolerance loosening, subsampling, reduced trees/draws, coarser grids, or stats-changing
shortcut was used. AR/RW/UCSV cached numbers were not recomputed by this command.

## 2026-07-09 -- B1 post-rebase UCSV prior-knob pass

### Verdict

Branch `repro/medeiros-2021` was rebased cleanly onto local `main` at
`c3cd17e5cd0ba9dfc851318ded436986d53bfecd`. No merge conflicts occurred.

Post-rebase Table 5 parity:

| Arm | h=1 ratio / paper / d | h=3 ratio / paper / d | h=6 ratio / paper / d | h=12 ratio / paper / d |
| --- | ---: | ---: | ---: | ---: |
| AR | 0.911304 / 0.902 / +0.009304 MATCH | 0.789546 / 0.790 / -0.000454 MATCH | 0.792246 / 0.791 / +0.001246 MATCH | 0.764836 / 0.753 / +0.011836 CLOSE |
| UCSV | 0.914794 / 0.954 / -0.039206 CLOSE | 0.716745 / 0.797 / -0.080255 DIVERGENT | 0.714645 / 0.777 / -0.062355 DIVERGENT | 0.695902 / 0.781 / -0.085098 DIVERGENT |
| RF | 0.859709 / 0.844 / +0.015709 CLOSE | 0.732985 / 0.706 / +0.026985 CLOSE | 0.749831 / 0.715 / +0.034831 CLOSE | 0.710726 / 0.685 / +0.025726 CLOSE |

RW baseline stayed `0.2889 / 0.3847 / 0.3852 / 0.3990` with OOS counts
`311 / 309 / 306 / 300`.

B1 is therefore **not fully replicated**: AR and RF are faithful, but UCSV remains
divergent at h=3/h=6/h=12 after applying the new paper-prior knobs.

### UCSV Parameter Map

Paper Appendix B.1 (`qa/medeiros_correct_fulltext.txt:1843-1852` after null stripping)
states the UCSV equations, MCMC estimation, one-sided `tau_t|t` h-step forecasts, and
`Vtau=Vh=0.12` initial-prior variances. The author repos contain no UCSV implementation:
targeted search over `qa/ForecastingInflation/*.R`, `qa/ForecastingInflation/functions/*.R`,
`qa/HDeconometrics/R`, and `qa/HDeconometrics/man` found no `UCSV`, `ucsv`, `Stock`,
`Watson`, `gamma`, `Vtau`, `Vh`, `0.12`, or `MCMC` implementation hit.

Runner mapping now uses the new package knobs:

```python
UCSV_PARAMS = {
    "gamma": 0.2,
    "initial_obs_log_vol_variance": 0.12,
    "initial_level_log_vol_variance": 0.12,
    "random_state": 42,
}
```

Interpretation:

- `initial_obs_log_vol_variance=0.12` and
  `initial_level_log_vol_variance=0.12` are the package-level expression of the paper's
  `Vtau=Vh=0.12` initial-prior variances.
- `gamma=0.2` remains the Stock-Watson log-volatility random-walk innovation variance
  exposed by the package. The paper text says the innovation variances use inverse-gamma
  priors but does not publish hyperparameters or a fixed macroforecast-compatible gamma.
- `n_draws=5000` and `burn=1000` remain package defaults. No draw reduction or tolerance
  relaxation was used.

Effect of the new UCSV knobs:

| Horizon | Previous UCSV | New-knob UCSV | Paper | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.911726 | 0.914794 | 0.954 | CLOSE |
| 3 | 0.726410 | 0.716745 | 0.797 | DIVERGENT |
| 6 | 0.720370 | 0.714645 | 0.777 | DIVERGENT |
| 12 | 0.695099 | 0.695902 | 0.781 | DIVERGENT |

The prior-variance mapping moved UCSV only slightly and did not close h=3/h=6/h=12.
The remaining gap is a genuine UCSV protocol/package caveat, most likely around the
paper's unpublished inverse-gamma variance-prior details and MCMC implementation choices
versus the package's fixed Stock-Watson/Kim-Shephard-Chib sampler interface.

### #3 / #4 Workaround Status

- `#3` UCSV prior knobs removed the earlier expressiveness blocker: the runner can now
  set the paper's `Vtau=Vh=0.12` through `initial_obs_log_vol_variance` and
  `initial_level_log_vol_variance`, and it sets a fixed `random_state=42`.
- `#4` horizon-dependent window support removed the runner-local `MedeirosRollingWindow`
  subclass. The runner now uses
  `mf.window.from_cutoffs(..., estimation_size=base,
  estimation_size_rule=medeiros_rolling_size)`.
- The `#1` RF `model_selection={"random_forest": None}` workaround was left in place as
  a harmless explicit guard. RF numbers were unchanged across the window swap, so this pass
  did not spend time removing it.

### Rerun / P4 Guard

Commands run:

```bash
python3 scripts/replication/medeiros_2021_pipeline/run_block.py --arms ar,rf
python3 scripts/replication/medeiros_2021_pipeline/run_block.py --arms ar,rf
python3 scripts/replication/medeiros_2021_pipeline/run_block.py --arms ucsv
python3 scripts/replication/medeiros_2021_pipeline/run_block.py
```

The first `--arms ar,rf` established the post-rebase pre-swap baseline; the second verified
the package window API swap. The window identity changed result-store keys, so these were
not pure cache-hit checks, but the selected RW/AR/RF numbers were byte-identical at printed
precision:

| Check | s1 sec | s2 sec | Total sec | Result |
| --- | ---: | ---: | ---: | --- |
| Pre-swap RW/AR/RF baseline | 285.2 | 546.3 | 831.5 | AR/RF close/match table unchanged |
| Post-swap RW/AR/RF equivalence | 288.0 | 548.7 | 836.7 | RW RMSE/OOS, AR ratios, RF ratios unchanged |
| UCSV-only new-prior rerun | 1448.6 | 2654.4 | 4103.0 | UCSV h=3/h=6/h=12 still divergent |
| Final cached all-arm scorer | 2.8 | 4.3 | 7.3 | Final parity table assembled from `qa/result_cells` |

No `macroforecast/**` package code was edited. No G3, CSR/JMA/LASSO, bagging, hybrids,
subsampling, reduced trees, reduced UCSV draws, coarser grids, or tolerance changes were
used. Result-store reuse worked for the final all-arm scoring pass and for RW during the
UCSV selected-arm pass after the new window keys existed.

### New Findings

- `[UCSV PROTOCOL GAP, high]` The new knobs close the expressiveness blocker but not the
  Table 5 UCSV parity gap. The package still cannot express the paper's inverse-gamma
  innovation-variance priors as described in Appendix B.1; it exposes a fixed `gamma`
  instead.
- `[RESULT-STORE IDENTITY NOTE, low]` Replacing the custom window subclass with the package
  `estimation_size_rule` changes the window echo used in result-store identities. This is
  statistically correct but not a pure cache-hit migration; the equivalence check had to
  recompute selected RW/AR/RF cells under the new window identity before the final cached
  scorer became cheap again.

## 2026-07-10 -- B1 UCSV forecast-extraction correctness diagnostic

### Verdict

**BUG in the B1 UCSV extraction path, not in the low-level `ucsv()` estimator.**

The requested `.dev-notes/REPLICATION_OBJECTIVES.md` file was absent in this worktree, as
noted in earlier B1 progress notes, so this diagnostic used the current task note plus the
local Medeiros findings/docs. No `macroforecast/**` package file was patched, no G3 or other
arms were run, and no gamma/draw/prior sweep was performed.

### Low-Level Estimator Quantity

`macroforecast/models/bayesian.py` has the correct low-level Stock-Watson point-forecast
quantity for the sample passed into it:

- `_UCSVForecaster.fit()` loops over MCMC draws, samples the UCSV trend, accumulates retained
  draws after `burn`, sets `self.trend_ = trend_sum / kept`, and sets
  `self.forecast_ = self.trend_[-1]` (`macroforecast/models/bayesian.py:90-123`).
- `_UCSVForecaster.predict()` returns `np.full(len(X), self.forecast_)`
  (`macroforecast/models/bayesian.py:125-126`), so a fitted estimator is horizon-invariant.
- `_sample_random_walk_state()` samples the final state directly from
  `Normal(filt_mean[-1], filt_var[-1])` before the backward smoothing recursion only fills
  earlier positions (`macroforecast/models/bayesian.py:237-264`). Thus the final draw is
  from the one-sided filtered final-state distribution for the fitted sample, not a
  lookahead-smoothed interior state.
- `ucsv()` exposes this in diagnostics as `trend` and `forecast`, with metadata
  `"forecast_is_final_trend": True` (`macroforecast/models/bayesian.py:305-329`).

Fixed-seed low-level toy check:

```text
origin sample end: 2004-01-01
forecast:                         2.990553584359
diagnostics["trend"].iloc[-1]:     2.990553584359
mean retained sampled tau_T draws: 2.990553584359
max abs diff across 12 predict rows: 0.0
full-sample smoothed-origin proxy at 2004-01-01: 6.422230182256
proxy minus origin-only forecast: 3.431676597897
```

Interpretation: for an origin-only fit, low-level `ucsv()` returns the posterior mean over
retained MCMC draws of `tau_{T|T}` and repeats it for every prediction row. Fitting the same
toy series with future post-origin observations gives a materially different origin trend,
so the low-level result is not the full-sample smoothed `tau_{T|n}`.

### B1 Extraction Defect

The current B1 runner arm is not extracting that paper forecast. The direct-policy path is:

- The B1 registry uses `Arm("ucsv", model="ucsv", features=None, params=UCSV_PARAMS)`
  (`scripts/replication/medeiros_2021_pipeline/registry.py:31-36` and the arm list below).
- For `features=None`, `_feature_spec_for_policy()` creates
  `feature_spec(target=target, horizon=h, target_mode="direct", target_transform=...)`
  (`macroforecast/forecasting/policy_config.py:115-160`).
- `direct_target()` sets the target at row `t` to `target[t+h]`
  (`macroforecast/feature_engineering/targets.py:41-65`).
- `forecast_direct_origin()` then filters training labels to
  `positions <= origin_pos - h` (`macroforecast/forecasting/policies/direct.py:54-67`,
  backed by `macroforecast/forecasting/selection_stage.py:51-67`).

This avoids realized target values after the origin: in the toy check below, the last kept
target value at every h equaled `y_T`, not `y_{T+h}`. But it still fits separate
horizon-shifted target sequences. For h=12, for example, the fitted target sequence is
aligned as `y_{t+12}` through `y_T`, not the same unshifted `y_1, ..., y_T` sequence used at
h=1. Therefore the B1 UCSV forecasts are not guaranteed to be the same value across h, even
though the paper's driftless-random-walk trend forecast must be flat across h.

Fixed-seed B1-style direct-policy toy check at origin `2004-02-01`:

```text
target-availability check:
h=1  last_label=2004-01-01  last_value=6.538057  origin_value=6.538057  future_value=6.660144
h=3  last_label=2003-11-01  last_value=6.538057  origin_value=6.538057  future_value=6.902468
h=6  last_label=2003-08-01  last_value=6.538057  origin_value=6.538057  future_value=7.153500
h=12 last_label=2003-02-01  last_value=6.538057  origin_value=6.538057  future_value=7.546306

pipeline direct-policy UCSV predictions from the same origin:
h=1   2.972313648151
h=3   6.527772055861
h=6   6.538136516755
h=12  4.939258595432
max-min across h: 3.565822868604
```

This is a minimal repro of the extraction bug: the current B1 direct-policy UCSV forecast is
not the paper's single one-sided `tau_{T|T}` reused for all horizons. The runner should fit
UCSV once per origin on the unshifted target sample available through the origin and copy
that scalar forecast to h=1/3/6/12, or provide a target-kind policy that gives
horizon-invariant state-space benchmarks the unshifted origin sample while preserving
target-date evaluation.

Fixing this would plausibly move the UCSV Table-5 row because the current extraction can
produce materially different h-specific forecasts before any paper-under-specification
issues are reached. After the extraction is fixed and rerun, any remaining gap can be
attributed to the paper's unpublished MCMC internals only if the flat one-sided forecast
still diverges.

## 2026-07-10 -- B1 UCSV flat one-sided extraction fix

### Resolution

The B1 runner now extracts UCSV as the paper's flat one-sided level-trend forecast:
fit the unshifted target through the forecast origin and reuse `tau_{T|T}` for
h=1/3/6/12.

Implementation in `scripts/replication/medeiros_2021_pipeline/run_block.py`:

- `ucsv_level_window()` gives UCSV a horizon-invariant rolling window with
  `R=regime_base - 1 - 4 - 1`: `354` in s1 and `486` in s2.
- `regime_arms()` applies that window only to `Arm("ucsv", model="ucsv", ...)`.
- `pipeline_spec(..., on_unsupported_direct="warn")` keeps RW on the original direct
  path, then the runner applies `PipelineSpec.policy_overrides={("ucsv",
  "CPIAUCSL"): "recursive"}` so only UCSV sees the non-shifted recursive/level path.
- UCSV remains a competitor. RW remains the sole `is_benchmark=True` denominator.
- No `macroforecast/**` package file was patched, and no UCSV post-processing was used.
  Forecast rows come from the package forecast path.

Classification: config fix in the replication runner using existing package execution
hooks. Small package/API caveat: the public `on_unsupported_direct="reroute"` switch is
too broad for this paper because it would reroute both `naive` and `ucsv`; a public
per-arm policy override would avoid the post-`pipeline_spec()` `dataclasses.replace(...)`
step.

### Flatness Check

Selected UCSV run output:

```text
UCSV_FLATNESS status=PASS common_origins=300 max_abs_range=0 forecast_policy=recursive
UCSV_FLATNESS_SAMPLE origin=1990-01-01 00:00:00 h=1:0.558519124429 h=3:0.558519124429 h=6:0.558519124429 h=12:0.558519124429
```

### Corrected Table 5 Parity

Full cached all-arm scorer after the UCSV extraction fix:

| Arm | h=1 ratio / paper / d | h=3 ratio / paper / d | h=6 ratio / paper / d | h=12 ratio / paper / d |
| --- | ---: | ---: | ---: | ---: |
| AR | 0.911304 / 0.902 / +0.009304 MATCH | 0.789546 / 0.790 / -0.000454 MATCH | 0.792246 / 0.791 / +0.001246 MATCH | 0.764836 / 0.753 / +0.011836 CLOSE |
| UCSV | 0.914794 / 0.954 / -0.039206 CLOSE | 0.729654 / 0.797 / -0.067346 DIVERGENT | 0.725196 / 0.777 / -0.051804 DIVERGENT | 0.697608 / 0.781 / -0.083392 DIVERGENT |
| RF | 0.859709 / 0.844 / +0.015709 CLOSE | 0.732985 / 0.706 / +0.026985 CLOSE | 0.749831 / 0.715 / +0.034831 CLOSE | 0.710726 / 0.685 / +0.025726 CLOSE |

RW baseline remained:

| h | OOS n | RW RMSE |
| ---: | ---: | ---: |
| 1 | 311 | 0.2889 |
| 3 | 309 | 0.3847 |
| 6 | 306 | 0.3852 |
| 12 | 300 | 0.3990 |

Verdict: B1 still does **not** fully replicate. AR and RF remain faithful/close. UCSV is
now definitionally flat across horizons, but h=3/h=6/h=12 remain outside the UCSV Table-5
tolerance after the extraction fix.

### Result-Store / P4 Guard

Commands run:

```bash
python3 scripts/replication/medeiros_2021_pipeline/run_block.py --arms ucsv
python3 scripts/replication/medeiros_2021_pipeline/run_block.py
```

Runtime:

| Run | s1 sec | s2 sec | Total sec | Result |
| --- | ---: | ---: | ---: | --- |
| UCSV-only flat extraction | 1600.3 | 2669.0 | 4269.3 | Recomputed changed UCSV recursive cells; RW reused for scoring |
| Final all-arm cached scorer | 2.8 | 4.2 | 7.1 | Assembled RW/AR/UCSV/RF from `qa/result_cells` |

Result-store manifest check: the newest written cells after the selected run are all
`arm=ucsv`, `target_policy=recursive` (four s2 cells at `2026-07-09T18:54Z` and four s1
cells at `2026-07-09T18:07-18:10Z`). No new AR/RF/RW cells were written in this pass.

P4 guard: no gamma/draw/burn/prior sweep, no reduced draws, no subsampling, no tolerance
change, no G3/CSR/JMA/LASSO/bagging/hybrid arms, and no `macroforecast/**` patch. AR/RF/RW
printed values are byte-identical to the prior cached all-arm table.

## [ENHANCEMENT, low — 2026-07-10] pca_step lacks an R-`princomp`-compatible mode
Found while verifying B1 RF factor composition. B1's RF (and the 5 ML arms) build factors via a
CUSTOM feature-step (`author_rf_embed` + `_r_princomp_scores`, registry.py:57-160) rather than the
built-in `feature_engineering.pca_step`, because:
- built-in `pca_step`: standard PCA (SVD/correlation, `scale=True`).
- author R `princomp(scale(df))`: COVARIANCE eigenvectors (`np.cov` + `eigh`) with a sign-flip
  convention (each loading's first element forced nonnegative), on the full in-window panel incl.
  the target, then embed(lags 0..3, R order) + Nov-2008 dummy.
The composition pattern (factors -> any model = PC-RF / PC-ridge/lasso/EN) IS fully supported via
`feature_spec([...]) + Arm(model=...)`; only the exact PCA *flavor* needed a custom step.
CANDIDATE: add a general option to `pca_step` — e.g. `decomposition={"svd"(default),"covariance_eigh"}`
+ `sign_convention={"none"(default),"first_positive"}` — so R-princomp-based paper replications can
compose PCA factors with any model without a custom step. General atomic-unit value (R princomp is a
common convention in published replication code); niche => LOW priority. Not a bug; no package patch made.
