# Hounyo and Li (2026) Replication Trust Note

Paper: Hounyo and Li, "Supervised Scaled Principal Component Analysis for Forecasting Using High-Dimensional Time Series", *International Journal of Forecasting* 42 (2026), 414-433.

The journal replication note reports that the numerical results were not reproduced, owing to computational cost. This page records the package-side trust result for the Table 2 macro/inflation smoke target: inflation, h=1, full sample, no threshold.

## Bottom Line

`macroforecast` is verified correct for the cheap Hounyo-Li Table 2 methods. The honest leak-free package surface gives different numbers because it does not look ahead. When the same package estimators are fed the author's look-ahead standardization surface in a labeled diagnostic runner path, the cheap Table 2 rows reproduce.

The author surface is not a package feature. It is exposed only through the replication runner as `--surface author_oracle`.

## Arm Map

All leak-free arms are pinned to the author method, not to package defaults.

| Paper arm | Package method | Leak-free configuration |
|---|---|---|
| AR_BIC | `ar_bic` | `min_lag=1`, `max_lag=12`, `criterion="bic"`, `ic_parameter_count="lag_square"`, `estimator="matlab_ar"`, `forecast_mode="coefficient_power"`, `include_constant=True`, `horizon=1` |
| PCA | `pcr` | `n_components=1..10`, target-lag control, constant, `drop_control_columns=True`, `standardize=True`, `nan_policy="zero_after_standardize"`, no quadratic factors |
| sPCA | `scaled_pca` | `n_components=1..10`, `scale=False`, target-lag control, constant, no quadratic factors |
| PLS | `pls` | `n_components=1..10`, `scale=False`, `score_projection="x_weights_raw"`, target-lag control, constant, no quadratic factors |
| SPCA | `supervised_pca` | `n_components=1..10`, `n_selected=18:6:108`, `scale=False`, `preselect="none"`, raw-before-standardize preselect support, target-lag control, constant |
| SsPCA | `supervised_scaled_pca` | Same supervised grid and controls as SPCA, with supervised scaled-PCA slope scaling |

Shared leak-free preprocessing: no transform, no outlier filter, zero imputation, z-score standardization with `ddof=1`, `standardize_scope="origin_available_predictors"`, and post-standardization non-finite values set to zero. The validation geometry follows the author fold boundaries `{80,130,190,240}` with fold-internal expanding refits and K/qN grids. The public A2 splitter is not defective; the remaining author/package difference is the surface being scored, not the fold mechanics.

## Cheap Table 2 Results

Command for the honest package surface:

`python3 scripts/replication/hounyo_li_2026_pipeline/run_g1_smoke.py --result-store runs/hl2026_store --n-jobs auto --parallel-cell-timeout 21600 --arms cheap`

Command for the labeled author-methodology diagnostic:

`python3 scripts/replication/hounyo_li_2026_pipeline/run_g1_smoke.py --surface author_oracle --result-store runs/hl2026_store --arms cheap`

AR_BIC is the denominator in both columns.

| Method | Paper Table 2 | macroforecast leak-free | Leak-free verdict | macroforecast on author-methodology surface | Author-surface verdict |
|---|---:|---:|---|---:|---|
| PCA | 0.970 | 1.081 | Honest leak-free result, not Table 2 | 0.970590 | Reproduced |
| sPCA | 0.768 | 0.965 | Honest leak-free result, not Table 2 | 0.768000 | Reproduced |
| PLS | 0.861 | 1.037 | Honest leak-free result, not Table 2 | 0.860446 | Reproduced |

The author-methodology reproduction table is written to `runs/hl2026_store/author_oracle_reproduction.csv`; forecasts and provenance are in `runs/hl2026_store/author_oracle_forecasts.csv` and `runs/hl2026_store/author_oracle_report.json`.

Author-surface mechanics: PCA final forecasts use package `pcr(standardize=False)`, and PLS final forecasts use package `pls(scale=False, score_projection="x_weights_raw")`. For sPCA, the runner first constructs the author's leaky `scaleXs` block from `inflation_linear.m:199-211`; after that external slope surface is fixed, the forecast is the same scaled-PCA factor-regression algebra. The slope leak is intentionally not exposed as a normal `scaled_pca` package option.

## The Leak

The author standardizes each 241-row target block before cross-validation and forecasting, including the realized `y_{T+h}`. In the MATLAB path this is the target block formed at `inflation_linear.m:192-197`; `wt` and `ytplush` are then built from that already-standardized block. The predictor block is also standardized once over the 240-row origin block before CV.

That look-ahead target standardization drives both the forecast and K selection. It changes the validation loss surface: on sampled origins, `macroforecast.pcr` on the author surface selects the author's K exactly, while the leak-free package surface selects different K values. It also changes the final forecast scale.

The key diagnostic is decisive: `macroforecast.pcr` is bit-identical to the author PCA algebra when fed the author's standardized predictor block, leaky target/control block, author split, and author K. The full author-methodology runner then reproduces PCA, sPCA, and PLS against the author AR_BIC denominator.

## Supervised Arms

SPCA and SsPCA are documented as a compute wall, not as a package limitation. The exact-author supervised run requires fold-internal expanding refits across the full `(K,qN)` grid. The measured exact configuration is greater than 2 hours per supervised cell for one target and one horizon; Table 2 alone needs 12 supervised cells, and the full paper run is weeks-scale.

The package can express the exact supervised leak-free configuration through `supervised_pca` and `supervised_scaled_pca`. Running the full author supervised look-ahead surface is intentionally not part of this B2 final check.

## Verdict

`macroforecast` is correct: `pcr` is bit-identical to the author algebra, and the A2 splitter is not defective. Hounyo-Li Table 2 depends on the author's look-ahead standardization surface. The package's honest leak-free numbers are the results users should trust; the labeled `author_oracle` runner path exists only to prove that the package estimators reproduce the paper when given the author's leaky surface.

B2 also left reusable package capabilities and configuration hooks, including `pcr`, `ar_bic`, raw-weight PLS scoring, and author-aligned factor/supervised settings. No `macroforecast/**` code is patched for the author look-ahead surface.
