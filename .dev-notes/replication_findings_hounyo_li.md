# Hounyo and Li 2026 B2 Replication Findings

Scope: B2 setup plus G1 smoke only. No G2/G3 table/full runs and no finance scope.

## Latest Cheap-Only Result

Branch `repro/hounyo-li-2026` is on `main` commit `3cebafc9`. No `macroforecast/**` files were patched.

The G1 smoke was rerun with the cheap arm subset:

`python3 scripts/replication/hounyo_li_2026_pipeline/run_g1_smoke.py --result-store runs/hl2026_store --n-jobs auto --parallel-cell-timeout 21600 --arms cheap`

Selected arms: `RW`, `AR_BIC`, `PCA`, `sPCA`, `PLS`. Supervised arms excluded by scope: `SPCA`, `SsPCA`.

Final warmed-cache runtime was 3.32 seconds. `n_jobs="auto"` resolved to five workers. Result store path was reused (`runs/hl2026_store`): `n_computed=0`, `n_reused=5`, `n_undigestible=0`, no failed cells, no empty cells.

Operational cache note: the first cheap-only invocation found no `RW` manifest in `runs/hl2026_store`; it computed `RW` once in 1,423.29 seconds while reusing the four corrected factor/AR cells. The immediate rerun then reused all five cheap cells.

| Method | Corrected smoke ratio | Paper Table 2 | Delta | Verdict |
|---|---:|---:|---:|---|
| AR_BIC | 1.000000 | 1.000 | 0.000000 | BENCHMARK |
| PCA | 1.080953 | 0.970 | 0.110953 | FAIL |
| sPCA | 0.964519 | 0.768 | 0.196519 | FAIL |
| PLS | 1.037423 | 0.861 | 0.176423 | FAIL |

`RW` is included in the cheap run but is not a Table 2 factor-method row. Its relative MSE versus `AR_BIC` was 2.383436 over 361 common targets.

Conclusion: the corrected cheap mappings did not close the Table 2 parity gap under the package-safe, leak-free target configuration. No completed factor method is within the +/-0.03 tolerance. The cheap-method rank among paper rows is `sPCA`, `PLS`, `PCA`, matching the paper's cheap-method rank, but levels remain too high.

## Mapping Confirmation

- `RW` uses target-only `model="naive"`.
- `AR_BIC` uses `model="ar_bic"` with `min_lag=1`, `max_lag=12`, `criterion="bic"`, `ic_parameter_count="lag_square"`, `estimator="matlab_ar"`, `forecast_mode="coefficient_power"`, `include_constant=True`, and `horizon=1`.
- `PCA` uses `model="pcr"` with target-lag control, constant, dropped control columns, `standardize=True`, `nan_policy="zero_after_standardize"`, no quadratic factors, and K grid `1..10`.
- `sPCA` uses `scaled_pca` with `scale=False`, target-lag control, constant, no quadratic factors, and K grid `1..10`.
- `SPCA` remains expressible as `supervised_pca` with `scale=False`, `preselect="none"`, `preselect_stage="raw_before_standardize"`, target-lag control, constant, and K/qN grid.
- `SsPCA` remains expressible as `supervised_scaled_pca` with the same supervised controls/grid.
- `PLS` uses `pls` with `score_projection="x_weights_raw"`.
- Fold CV uses `score_aggregation="mean_fold"`.
- Predictor preprocessing uses `standardize_scope="origin_available_predictors"` and a local non-finite-to-zero step after standardization.
- The author's target-y standardization leak was not reproduced.

## Movement Versus Old Mapping

Old completed-method smoke used `PCA -> far`, old sklearn-transform PLS scores, and `AR_BIC -> ar` plus IC search. Compared with the old numbers requested in the B2 cheap task:

| Method | Old ratio | Corrected ratio | Paper | Old abs gap | New abs gap | Movement |
|---|---:|---:|---:|---:|---:|---|
| PCA | 1.031 | 1.080953 | 0.970 | 0.061000 | 0.110953 | Moved away |
| sPCA | 0.901 | 0.964519 | 0.768 | 0.133000 | 0.196519 | Moved away |
| PLS | 1.112 | 1.037423 | 0.861 | 0.251000 | 0.176423 | Moved toward, still FAIL |

`AR_BIC` has no independent Table 2 factor ratio because it is the denominator. The denominator is now API-aligned with the requested `ar_bic` backend/options, but that did not make the completed ratios match the paper.

## Residual Attribution

The remaining completed-method deltas are +0.110953 for PCA, +0.196519 for sPCA, and +0.176423 for PLS. These are directionally consistent with not reproducing the author's leaky full-block target standardization: the paper's target-y leak can make pseudo-OOS performance look better than a package-safe forecast.

The known AR denominator diagnosis quantified one target-surface gap at about MSE ratio 1.059638 (`our/author`) for the AR benchmark. That AR-only magnitude is not enough by itself to explain the factor-method residuals. Applying that denominator factor to the current ratios would produce 1.145419 for PCA, 1.022041 for sPCA, and 1.099293 for PLS, farther from the paper. The unreplicated factor-side target-y leak and rolling-block target surface remain the main residual attribution, but this run does not prove that the target-y leak alone would close parity.

## Supervised Compute Wall

The exact supervised cells are a compute wall under the author fold-internal expanding-refit x `(K,qN)` grid. The prior full-smoke attempt completed `AR_BIC`, `PCA`, `sPCA`, and `PLS`, but `SPCA` and `SsPCA` each timed out after 21,600 seconds with `ParallelExecutorTimeout: no cell completed within 21600.0 seconds`. The measured cost is greater than 2 hours for one supervised cell at one target and one horizon.

Table 2 requires 12 supervised cells; the full paper replication would take weeks. This aligns with the paper's own footnote: "numerical results were not reproduced, owing to the substantial computational cost". The package can express the exact leak-free supervised configuration (`SPCA` as `supervised_pca`, `SsPCA` as `supervised_scaled_pca`), so the exact-author full run is a documented compute wall, not a package limitation. A smaller supervised feasibility run with reduced grid or origins would not be the exact author result and should only be run as a separately labeled optional exercise.

## New Gaps / Fix-Lane Items

- B2 PCA K-selection diagnosis (2026-07-11): the dominant author/package K gap is not a public `explicit_folds(..., within_fold="expanding")` refit-cadence bug. On sampled origins, feeding `macroforecast.pcr` the exact author `PCA_tune.m` surface (predictors standardized once over the full 240-row origin block; target/control standardized once over the 241-row target block; `pcr.standardize=False`) reproduces the author K argmin exactly. The current B2 runner instead selects on the package-safe raw target/control surface with in-fit predictor standardization (`run_g1_smoke.py:211-218`, `inflation_linear_tune.m:187-195`, `PCA_tune.m:17-101`). Exact-author replication therefore needs an explicit author-oracle/leaky target-standardization lane in the B2 runner, not a `macroforecast/**` patch.
- Secondary fold-CV API caveat from the same diagnosis: current B2 uses callable `_author_expanding_folds_or_closest` (`run_g1_smoke.py:145-191`) to clip the first origin, but callable splitters are assigned `fold_ids = range(len(splits))` in `macroforecast/model_selection/splitters.py:61-71`. Consequently `score_aggregation="mean_fold"` in `macroforecast/model_selection/runner.py:51-55` averages one-observation split MSEs for this callable, not three logical fold MSEs. Public `explicit_folds` preserves fold IDs (`splitters.py:122-143`), so the leak-free fix lane is either use `explicit_folds([80,130,190,240], within_fold="expanding")` where 240 rows exist and handle the first origin separately, or add a general fold-id-preserving callable splitter API.
- High for exact author-oracle parity: the package pipeline has one shared target/evaluation surface. It cannot cleanly express AR_BIC's separate caller-side `diff + movmean(12)` target preparation while evaluating factor-method forecasts on their own target surface against the same benchmark. A future fix lane needs an explicit opt-in author-oracle/per-arm target-surface mode or externally supplied benchmark forecasts. This branch did not patch the package.
- Medium: top-level `preprocess_spec` does not currently expose a direct post-standardization non-finite fill knob for the predictor standardization step. The runner used a local custom step to implement the requested zero-after-standardize behavior without patching `macroforecast/**`.
- Medium: `ar_bic` and `naive` under the direct Table 2 forecast policy require `on_unsupported_direct="warn"` because the package guard does not know that this h=1 author smoke deliberately accepts these target-only dynamics. The opt-out is recorded in the runner; no package patch was made.
- Operational: the result store did not initially contain `RW` despite the task note. `RW` was computed once and is now cached; the final warmed run reused all five cheap cells.

## Setup Confirmations

- Author package was listed before extraction and only the macro/inflation pieces were extracted to `qa/hounyo_li_matlab/`.
- Data source is the author macro workbook, converted by `scripts/replication/hounyo_li_2026_pipeline/build_data.py`.
- Converted author panel files:
  - `qa/hounyo_li_macro_panel.csv`
  - `qa/hounyo_li_inflation.csv`
  - `qa/hounyo_li_macro_inflation_panel.csv`
  - `qa/hounyo_li_author_data_manifest.json`
- Manifest check: merged author data has 721 monthly rows, 126 predictors plus `CPIAUCSL`; the h=1 smoke slice has 601 rows from 1973-03-01 through 2023-03-01, yielding 361 h=1 OOS forecast targets for 1993-03-01 through 2023-03-01 under the package direct-date convention.

## Efficiency / Fidelity Notes

- The cheap-only run is a scope reduction, not a fidelity change to the methods that ran.
- No reduced grid, reduced origin set, reduced method shortcut, finance/G2/G3 scope, or target-y leak was introduced.
- Supervised `SPCA` and `SsPCA` are documented as a compute wall and excluded from the cheap-only scoring run.
- BLAS/OpenMP thread caps are set in the runner before importing numerical libraries to prevent worker oversubscription.
- Output files:
  - `runs/hl2026_store/g1_smoke_accuracy_raw.csv`
  - `runs/hl2026_store/g1_smoke_parity.csv`
  - `runs/hl2026_store/g1_smoke_report.json`
