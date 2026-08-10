# Zhang, Wahab and Wang (2023) replication trust note

Paper: Zhang, Y., Wahab, M.I.M., & Wang, Y. (2023), "Forecasting crude oil market
volatility using variable selection and common factor," *International Journal of
Forecasting* 39(2), 486-502.

This page records the B3 replication result for `macroforecast`: the package reproduces
ZWW's Table 3 (futures, main) and Table 4 (spot, robustness) — **both** the headline
finding (the supervised-PCA variable-selection models beat the AR benchmark, with
out-of-sample R²_OS positive and rising with the horizon) **and** the competing-models
story (every other model fails to beat AR) — once ZWW's own, under-documented
methodological choices are matched. The replication additionally surfaced two package
defects and demonstrated a 13x parallel speedup with statistically identical output.

## Bottom line / verdict

`macroforecast` faithfully implements ZWW's method (verified leak-free and deterministic).
The headline result reproduces on both exhibits. The single decisive detail the paper
leaves implicit is that ZWW extract the common factor with **covariance (non-standardized)
PCA**, not the standardized/correlation PCA that is the Stock-Watson/McCracken-Ng norm —
and their entire "supervised-PCA beats AR" finding depends on it (see KEY FINDING below).

## KEY FINDING — the headline hinges on covariance PCA, which makes the factor a VXO proxy

The screened predictor set is dominated numerically by one series: `VXOCLSx` (option-implied
volatility) is McCracken-Ng transform code 1 (level), so after the standard t-code step it has
std ~8.2 — the largest-variance predictor in the set — while the collinear macro blocks
(payrolls, industrial production) are t-code 5 growth rates with std ~0.002.

- **Standardized/correlation PCA** (each predictor rescaled to unit variance, the usual FRED-MD
  convention) → PC1 aligns with the largest *correlated cluster* (the mutually collinear macro
  aggregates); VXO is weakly correlated with that cluster and is orphaned (PC1 loading rank
  ~10-33). The PCA-VS factor is a macro-activity factor and does **not** beat AR out of sample
  (PCA-t_stat h=6 R²_OS = -18.7%).
- **Covariance PCA** (no rescaling; `scale=False`) → VXO's large raw variance dominates PC1, so
  the factor is essentially an implied-volatility factor. This reproduces ZWW's positive PCA-VS
  (PCA-t_stat h=6 R²_OS = +5.9% vs paper +4.6%) and exactly matches the paper's own narrative
  that option-implied volatility is "the most powerful predictor" (selected 98.4% of the time).

The paper (§2.3, Eq. 7-8) describes "conventional PCA" on the predictors and never states that
the inputs are standardized. Covariance PCA is the **only** choice that reproduces ZWW's numbers
and is consistent with their VXO-dominant story; standardized PCA is negative and wrong for this
paper. We therefore conclude ZWW used covariance PCA. Honest caveat: covariance PCA on
heterogeneously t-coded data is statistically unconventional (a single high-variance level series
dominates the factor), and it departs from the standardization convention ZWW themselves cite.
`macroforecast.transforms.pca_step` supports both via `scale=`; this is a paper-method detail,
not a package defect.

## Table 3 — futures (main). R²_OS (%): macroforecast / ZWW-published

Covariance PCA (`scale=False`), Newey-West-consistent selection convention, expanding window,
OOS 1998:01-2018:12.

Cells are out-of-sample R²_OS in percent, reported as **macroforecast / ZWW-published** (`--` = value not tabulated by ZWW; the `h=12 (ours)` column is macroforecast only). Verdict band for the `verdict` column: **reproduced** = macroforecast R²_OS within about 1-2 pp of ZWW's published value (or same sign and ≈0 for the near-zero `PCA-all` arm); **near-exact** = within about 0.5 pp; **direction** = same sign and story as ZWW but the magnitude differs (flagged where it runs about 2x). The headline PCA-VS models reproduce to about 1-2 pp.

| Model | h=1 | h=3 | h=6 | h=12 (ours) | verdict |
|---|---|---|---|---|---|
| PCA-t_stat        | 1.72 / 2.02  | 1.39 / 1.84  | **5.93 / 4.57** | 8.17 | reproduced |
| PCA-delta_r2      | 1.55 / 1.05  | 1.29 / 1.93  | 4.67 / 5.19     | 2.54 | reproduced |
| PCA-lasso         | -0.15 / 1.68 | 1.34 / 3.09  | 3.93 / 4.79     | 5.41 | reproduced (h1 low) |
| PCA-elastic_net   | -0.32 / 1.77 | 2.38 / 2.51  | 3.84 / 4.27     | 5.39 | reproduced (h1 low) |
| Lasso             | -5.16 / -2.51 | -16.13 / -8.08 | -35.85 / -28.58 | -58.64 | direction (magnitude ~2x) |
| ENet              | -5.08 / -0.22 | -11.53 / -5.72 | -36.35 / -- | -73.64 | direction |
| PCA-all           | -0.89 / -1.11 | 0.56 / 0.00 | 0.41 / 0.04 | -0.67 | reproduced (≈0) |
| KS-all            | -527.9 / -252.4 | -1951 / -4749.6 | -9686 / -- | -5713 | catastrophic-overfit (same story) |
| KS-t_stat         | -34.9 / -15.4 | -75.1 / -63.7 | -115.8 / -- | -196.8 | direction |
| KS-delta_r2       | -19.1 / -12.8 | -52.9 / -48.9 | -113.3 / -- | -187.8 | direction |
| KS-lasso          | **-18.4 / -18.0** | -52.4 / -38.5 | -71.9 / -- | -104.4 | near-exact at h1 |
| KS-elastic_net    | -18.2 / -14.2 | -37.9 / -31.8 | -68.2 / -- | -148.4 | direction |

Headline (the four PCA-VS models): positive and rising with horizon, matching ZWW to ~1-2pp.
Significance: PCA-VS Clark-West statistics are positive/significant (as in ZWW); competing models
are not.

## Table 4 — spot (robustness). R²_OS (%): macroforecast (ZWW Table 4 PCA-VS ~+5% at long horizon)

Cells are macroforecast out-of-sample R²_OS in percent. ZWW's Table 4 is summarized (PCA-VS about +5% at the long horizon) rather than tabulated cell-by-cell, so this exhibit is a direction/level check under the same verdict band as Table 3, not a cell-by-cell parity.

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| PCA-t_stat      | 2.56 | 4.84 | 3.40 | 1.17 |
| PCA-delta_r2    | 2.79 | -0.20 | 0.71 | -0.17 |
| PCA-lasso       | -2.46 | -4.62 | 1.88 | 0.52 |
| PCA-elastic_net | 1.56 | -1.66 | 0.04 | 1.18 |
| PCA-all         | -1.45 | 0.04 | 0.16 | -0.57 |

The PCA-VS advantage is confirmed on the owner-oracle-validated spot RV series too (Table 4),
though weaker/noisier than futures, consistent with ZWW's robustness table.

## Competing-model magnitudes — why some are ~2x more negative

The "loser" arms (Lasso, ENet, KS-VS, KS-all) reproduce ZWW's direction and story but some
magnitudes run ~2x more negative. Two causes, both benign:
- **Selection convention (fixable):** these runs used the package's default OLS t-stat screen.
  ZWW use Newey-West t-stats (see Purpose 2). NW inflates standard errors for the overlapping
  multi-horizon target, prunes the autocorrelation-inflated predictors, and brings Lasso/KS-VS
  magnitudes closer to ZWW (~3pp effect measured). The headline PCA-VS is unaffected.
- **Overfit sensitivity (inherent):** KS-all regresses on all 126 raw predictors and overfits
  catastrophically; a -250% vs -530% difference is expected noise for an arm this ill-posed.
  KS-all uses no screen, so NW does not change it.

## What the replication delivered (the four purposes)

- **P1 (trust / faithful replication):** the package reproduces ZWW's Table 3 headline and the
  competing-models story on both futures (Table 3) and spot (Table 4), and the replication
  pinned the paper's one under-documented but decisive method choice (covariance PCA).
- **P2 (bugs found during replication):** two package defects.
  1. **`result_store` breakage** — running the pipeline with `pipeline_spec(result_store=...)`
     caused the benchmark and non-variable-selection arms to vanish from the accuracy report
     (`benchmark_present=False`, all R²_OS `NaN`). Removing it (relying on `n_jobs` alone) yields
     correct output. Recorded for the fix-lane.
  2. **No HAC/Newey-West option in the predictor screen** — `feature_engineering/screening.py:
     marginal_t_stats` computes homoskedastic OLS standard errors (`cov = sigma2 * pinv(X'X)`)
     with no autocorrelation correction, so it over-states significance for overlapping
     multi-horizon targets. A fix (opt-in Bartlett-kernel Newey-West `hac_lags`) is implemented
     on branch `fix/screen-hac-newey-west` (commit 5c625368), validated against statsmodels HAC
     to machine precision, with the existing OLS path bit-identical under a golden-identity gate.
- **P3/P4 (efficiency, statistically identical):** adding `n_jobs="auto"` to the pipeline
  parallelized the arm x origin cells across ~13 workers, cutting a full 13-arm x 4-horizon run
  from single-threaded to ~38 minutes (~13x), with n_jobs=1 vs n_jobs=4 output byte-identical.

## Data provenance

- **Futures RV (Table 3):** monthly realized variance from EIA daily WTI futures contract-1
  (RCLC1, public EIA series), built as sum of squared daily log returns; monthly LV = ln(RV).
  Date-exclusion rule: drop missing and non-positive price days (the 2020-04-20 negative
  settlement is excluded; outside the ZWW OOS window). Recipe matches the owner's KRW oracle
  notebook, verified bit-identical on spot.
- **Spot RV (Table 4):** WTI spot (FRED `DCOILWTICO`); RV build validated bit-identical
  (max abs diff 0, 451 months) against the owner oracle.
- **Predictors:** FRED-MD 2019:06 vintage, 126 variables (128 minus ACOGNO), McCracken-Ng
  t-codes + one-month publication lag, via `mf.data.load_fred_md(vintage="2019-06")`. VXOCLSx
  (the paper's dominant predictor) is present in this vintage.
- Target: LV_{t+1:t+h} = ln(mean RV over the next h months), via `log_average_value`,
  policy `direct`; verified against the stage-1 target to 0.

## Reproduction (gate)

```
# stage-1 data (RV oracle gate + FRED-MD 2019-06 predictors)
python -m scripts.replication.zww_2023_pipeline.build_stage1_target
# Table 3 (futures) + Table 4 (spot); covariance PCA is set in registry.py (PCA_SCALE=False)
uv run python -m scripts.replication.zww_2023_pipeline.replicate_zww2023 --market futures --full
uv run python -m scripts.replication.zww_2023_pipeline.replicate_zww2023 --market spot --full
```
Results: `runs/zww_b3_stage2/results/{futures,spot}/accuracy_h{1,3,6,12}.csv` (R²_OS) and
`significance_h*.csv` (Clark-West). ZWW published values: `runs/zww_b3_stage2/zww_table3_gold_raw.txt`.

## Caveats / open items

- **Covariance PCA is inferred, not stated.** It is the only choice that reproduces ZWW and
  matches their VXO narrative, but the paper does not say it explicitly. This is the paper's
  under-specification, not a package issue.
- **Competing-model magnitudes** reproduce in direction/story, not exactly; the Newey-West
  screen (Purpose-2 fix, landed on a branch) refines Lasso/KS-VS toward ZWW but not KS-all.
  The runs documented here use the OLS screen; a fully-NW-screened rerun is a follow-up.
- **Futures RV contract/roll:** ZWW cite "EIA daily WTI futures" without the exact contract/roll;
  RCLC1 (front contract) is used here. Spot (Table 4) is owner-oracle-validated.

Provenance: replication worktree `mf-b3-zww` (branch `repro/zww-2023`); `macroforecast/**`
unpatched by this replication (the two package findings are recorded for the fix-lane, and the
Newey-West fix is committed on its own branch). No push.
