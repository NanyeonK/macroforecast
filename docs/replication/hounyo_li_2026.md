# Hounyo and Li (2026) replication trust note

Paper: Hounyo and Li, "Supervised Scaled Principal Component Analysis for Forecasting Using High-Dimensional Time Series", *International Journal of Forecasting* 42 (2026), 414-433.

The paper's own replication note says: "The numerical results presented in this manuscript were not reproduced, owing to the substantial computational cost involved." This page records the B2 Table 2 trust result for `macroforecast`. The package can reproduce the author's published methodology when the author's look-ahead standardization surface is emulated. Its normal leak-free output differs for identified reasons.

## ⚠️ KEY FINDING - Look-ahead bias in the paper's target standardization

The headline finding is a look-ahead leak in the paper's out-of-sample evaluation. The author's MATLAB standardizes the target block before the forecast split and includes the realized future target `y_{T+h}` in that standardized block. In `inflation_linear.m:196-197`, the code forms `ytplush(:,1+h:T+h)` after standardizing the full target/control block. That realized future value is not available at a real forecast origin.

The reproduction evidence is tight. On the matched inflation h=1 PCA case, emulating the author's leaky surface reproduces Table 2: `macroforecast.pcr` gives 0.970590, matching the paper's 0.970 after rounding. The leak-free package-side diagnostic gives 1.080953 instead. The 0.110953 gap decomposes into about 0.0206 from the direct target-y standardization leak, or about 19%, and about 0.0904 from the author K/window surface, or about 81%, mainly K selection tuned on the leaky standardized surface. When `macroforecast.pcr` is fed the author-standardized X block, author leaky target block, author split, and author K, it is bit-identical to the author PCA code, with maximum absolute prediction difference `3.844e-14`, and it reproduces the author K choices in the diagnostic origins.

The implication is methodological, not accusatory. Based on the published MATLAB and cross-validated package checks, the paper's pseudo-OOS factor-method results are optimistically biased by a common but subtle form of target-standardization leakage. `macroforecast` is leak-free by design, so its honest out-of-sample numbers do not equal the paper's leaky Table 2 numbers.

## Verdict / Bottom line

The B2 checks verify `macroforecast` on the relevant implementation surfaces. `pcr` is bit-identical to the author algebra on the author surface. The A2 splitter diagnosis did not find a public splitter defect. The K-prefix grouped evaluator is bitwise-identical to the non-prefix path, with `max_forecast_abs_diff = 0.0` and `max_score_abs_diff = 0.0`.

The difference from Table 2 is not a package defect. It is the paper's look-ahead target standardization, plus the data-pipeline and COVID-period interaction documented below. The package reproduces Table 2 on the author's methodology in the locked B2 author-oracle grid: 55/60 cells within `|Delta| <= 0.03`, with inflation 20/20 exact. Its honest leak-free output differs for identified, documented reasons.

## Demonstration A - Package reproduction on the author methodology

The labeled author-oracle run uses the author's Table 2 macro PC/no-threshold surface. It covers inflation, IP growth, and unemployment; horizons 1, 6, 12, and 24; and the PCA, SPCA-family, and PLS rows. Ratios are scored against the local `AR_BIC` denominator. This is a diagnostic reproduction surface, not a normal package feature.

The locked B2 author-oracle grid completed all 60 cells and reproduced 55/60 within `|Delta| <= 0.03`. All 20 inflation cells pass. The wall runtime was 15,570.74 seconds, or 4:19:30, on 48 logical cores. The run used the verified K-prefix speedup; a naive supervised run at roughly 19.6 hours per cell would be weeks-scale.

| Target | h | Method | Reproduced | Paper | Delta | Verdict |
|---|---:|---|---:|---:|---:|---|
| Inflation | 1 | PCA | 0.970590 | 0.970 | +0.000590 | PASS |
| Inflation | 1 | SPCA | 0.823163 | 0.823 | +0.000163 | PASS |
| Inflation | 1 | sPCA | 0.768000 | 0.768 | -0.000000 | PASS |
| Inflation | 1 | SsPCA | 0.738333 | 0.738 | +0.000333 | PASS |
| Inflation | 1 | PLS | 0.860446 | 0.861 | -0.000554 | PASS |
| Inflation | 6 | PCA | 0.928317 | 0.928 | +0.000317 | PASS |
| Inflation | 6 | SPCA | 0.912252 | 0.912 | +0.000252 | PASS |
| Inflation | 6 | sPCA | 0.855733 | 0.855 | +0.000733 | PASS |
| Inflation | 6 | SsPCA | 0.848056 | 0.848 | +0.000056 | PASS |
| Inflation | 6 | PLS | 1.087046 | 1.082 | +0.005046 | PASS |
| Inflation | 12 | PCA | 1.076440 | 1.076 | +0.000440 | PASS |
| Inflation | 12 | SPCA | 1.049114 | 1.049 | +0.000114 | PASS |
| Inflation | 12 | sPCA | 0.959354 | 0.959 | +0.000354 | PASS |
| Inflation | 12 | SsPCA | 0.982966 | 0.983 | -0.000034 | PASS |
| Inflation | 12 | PLS | 1.199572 | 1.208 | -0.008428 | PASS |
| Inflation | 24 | PCA | 0.987448 | 0.987 | +0.000448 | PASS |
| Inflation | 24 | SPCA | 0.953026 | 0.953 | +0.000026 | PASS |
| Inflation | 24 | sPCA | 0.901740 | 0.902 | -0.000260 | PASS |
| Inflation | 24 | SsPCA | 0.858460 | 0.858 | +0.000460 | PASS |
| Inflation | 24 | PLS | 1.181597 | 1.180 | +0.001597 | PASS |
| IP growth | 1 | PCA | 1.126911 | 1.148 | -0.021089 | PASS |
| IP growth | 1 | SPCA | 0.967862 | 0.902 | +0.065862 | MISS |
| IP growth | 1 | sPCA | 1.045666 | 1.071 | -0.025334 | PASS |
| IP growth | 1 | SsPCA | 0.852153 | 0.844 | +0.008153 | PASS |
| IP growth | 1 | PLS | 1.164690 | 1.219 | -0.054310 | MISS |
| IP growth | 6 | PCA | 0.929859 | 0.930 | -0.000141 | PASS |
| IP growth | 6 | SPCA | 0.875332 | 0.903 | -0.027668 | PASS |
| IP growth | 6 | sPCA | 0.925957 | 0.923 | +0.002957 | PASS |
| IP growth | 6 | SsPCA | 0.877782 | 0.886 | -0.008218 | PASS |
| IP growth | 6 | PLS | 0.952630 | 0.948 | +0.004630 | PASS |
| IP growth | 12 | PCA | 1.023453 | 1.025 | -0.001547 | PASS |
| IP growth | 12 | SPCA | 1.016764 | 1.017 | -0.000236 | PASS |
| IP growth | 12 | sPCA | 0.969108 | 0.972 | -0.002892 | PASS |
| IP growth | 12 | SsPCA | 0.949870 | 0.984 | -0.034130 | MISS |
| IP growth | 12 | PLS | 1.056632 | 1.054 | +0.002632 | PASS |
| IP growth | 24 | PCA | 1.102544 | 1.107 | -0.004456 | PASS |
| IP growth | 24 | SPCA | 1.052497 | 1.055 | -0.002503 | PASS |
| IP growth | 24 | sPCA | 1.047703 | 1.045 | +0.002703 | PASS |
| IP growth | 24 | SsPCA | 1.007997 | 1.000 | +0.007997 | PASS |
| IP growth | 24 | PLS | 1.144298 | 1.149 | -0.004702 | PASS |
| Unemployment | 1 | PCA | 1.677038 | 1.644 | +0.033038 | MISS |
| Unemployment | 1 | SPCA | 1.626469 | 1.628 | -0.001531 | PASS |
| Unemployment | 1 | sPCA | 1.653358 | 1.654 | -0.000642 | PASS |
| Unemployment | 1 | SsPCA | 1.407908 | 1.411 | -0.003092 | PASS |
| Unemployment | 1 | PLS | 1.547527 | 1.698 | -0.150473 | MISS |
| Unemployment | 6 | PCA | 0.827417 | 0.825 | +0.002417 | PASS |
| Unemployment | 6 | SPCA | 0.803814 | 0.806 | -0.002186 | PASS |
| Unemployment | 6 | sPCA | 0.800449 | 0.798 | +0.002449 | PASS |
| Unemployment | 6 | SsPCA | 0.752633 | 0.766 | -0.013367 | PASS |
| Unemployment | 6 | PLS | 0.824370 | 0.831 | -0.006630 | PASS |
| Unemployment | 12 | PCA | 0.848229 | 0.849 | -0.000771 | PASS |
| Unemployment | 12 | SPCA | 0.824291 | 0.815 | +0.009291 | PASS |
| Unemployment | 12 | sPCA | 0.802197 | 0.802 | +0.000197 | PASS |
| Unemployment | 12 | SsPCA | 0.789784 | 0.778 | +0.011784 | PASS |
| Unemployment | 12 | PLS | 0.850109 | 0.849 | +0.001109 | PASS |
| Unemployment | 24 | PCA | 0.843002 | 0.842 | +0.001002 | PASS |
| Unemployment | 24 | SPCA | 0.808109 | 0.800 | +0.008109 | PASS |
| Unemployment | 24 | sPCA | 0.817001 | 0.812 | +0.005001 | PASS |
| Unemployment | 24 | SsPCA | 0.794135 | 0.785 | +0.009135 | PASS |
| Unemployment | 24 | PLS | 0.851229 | 0.853 | -0.001771 | PASS |

## Demonstration B - Honest leak-free output from `load_fred_md`

The normal package run uses `mf.data.load_fred_md`, FRED-MD vintage `2023-04`, origin-available predictor standardization, no future target values in scaling, model selection, or final fit, and `AR_BIC` using target history through the forecast origin only. This is the honest package output. It differs from the paper.

| Target | h | Method | Leak-free ratio | Paper | Delta |
|---|---:|---|---:|---:|---:|
| Inflation | 1 | PCA | 1.035999 | 0.970 | +0.065999 |
| Inflation | 1 | SPCA | 1.042119 | 0.823 | +0.219119 |
| Inflation | 1 | sPCA | 0.966038 | 0.768 | +0.198038 |
| Inflation | 1 | SsPCA | 1.129942 | 0.738 | +0.391942 |
| Inflation | 1 | PLS | 1.052079 | 0.861 | +0.191079 |
| Inflation | 6 | PCA | 1.012658 | 0.928 | +0.084658 |
| Inflation | 6 | SPCA | 1.015724 | 0.912 | +0.103724 |
| Inflation | 6 | sPCA | 1.044100 | 0.855 | +0.189100 |
| Inflation | 6 | SsPCA | 1.078240 | 0.848 | +0.230240 |
| Inflation | 6 | PLS | 1.231936 | 1.082 | +0.149936 |
| Inflation | 12 | PCA | 0.989559 | 1.076 | -0.086441 |
| Inflation | 12 | SPCA | 1.039798 | 1.049 | -0.009202 |
| Inflation | 12 | sPCA | 1.171296 | 0.959 | +0.212296 |
| Inflation | 12 | SsPCA | 1.274529 | 0.983 | +0.291529 |
| Inflation | 12 | PLS | 1.265209 | 1.208 | +0.057209 |
| Inflation | 24 | PCA | 1.007972 | 0.987 | +0.020972 |
| Inflation | 24 | SPCA | 1.050728 | 0.953 | +0.097728 |
| Inflation | 24 | sPCA | 1.282653 | 0.902 | +0.380653 |
| Inflation | 24 | SsPCA | 1.439747 | 0.858 | +0.581747 |
| Inflation | 24 | PLS | 1.281290 | 1.180 | +0.101290 |
| IP growth | 1 | PCA | 1.408970 | 1.148 | +0.260970 |
| IP growth | 1 | SPCA | 1.466397 | 0.902 | +0.564397 |
| IP growth | 1 | sPCA | 1.897550 | 1.071 | +0.826550 |
| IP growth | 1 | SsPCA | 1.488767 | 0.844 | +0.644767 |
| IP growth | 1 | PLS | 1.474617 | 1.219 | +0.255617 |
| IP growth | 6 | PCA | 1.045327 | 0.930 | +0.115327 |
| IP growth | 6 | SPCA | 1.064880 | 0.903 | +0.161880 |
| IP growth | 6 | sPCA | 1.131712 | 0.923 | +0.208712 |
| IP growth | 6 | SsPCA | 1.191232 | 0.886 | +0.305232 |
| IP growth | 6 | PLS | 1.082032 | 0.948 | +0.134032 |
| IP growth | 12 | PCA | 0.994982 | 1.025 | -0.030018 |
| IP growth | 12 | SPCA | 1.052672 | 1.017 | +0.035672 |
| IP growth | 12 | sPCA | 1.119861 | 0.972 | +0.147861 |
| IP growth | 12 | SsPCA | 1.128562 | 0.984 | +0.144562 |
| IP growth | 12 | PLS | 1.064709 | 1.054 | +0.010709 |
| IP growth | 24 | PCA | 1.067046 | 1.107 | -0.039954 |
| IP growth | 24 | SPCA | 1.022725 | 1.055 | -0.032275 |
| IP growth | 24 | sPCA | 1.087593 | 1.045 | +0.042593 |
| IP growth | 24 | SsPCA | 1.054378 | 1.000 | +0.054378 |
| IP growth | 24 | PLS | 1.097561 | 1.149 | -0.051439 |
| Unemployment | 1 | PCA | 4.863942 | 1.644 | +3.219942 |
| Unemployment | 1 | SPCA | 4.564064 | 1.628 | +2.936064 |
| Unemployment | 1 | sPCA | 3.573748 | 1.654 | +1.919748 |
| Unemployment | 1 | SsPCA | 4.347118 | 1.411 | +2.936118 |
| Unemployment | 1 | PLS | 4.925570 | 1.698 | +3.227570 |
| Unemployment | 6 | PCA | 1.311098 | 0.825 | +0.486098 |
| Unemployment | 6 | SPCA | 1.376535 | 0.806 | +0.570535 |
| Unemployment | 6 | sPCA | 1.431371 | 0.798 | +0.633371 |
| Unemployment | 6 | SsPCA | 1.391724 | 0.766 | +0.625724 |
| Unemployment | 6 | PLS | 1.567373 | 0.831 | +0.736373 |
| Unemployment | 12 | PCA | 1.079954 | 0.849 | +0.230954 |
| Unemployment | 12 | SPCA | 1.045745 | 0.815 | +0.230745 |
| Unemployment | 12 | sPCA | 1.087001 | 0.802 | +0.285001 |
| Unemployment | 12 | SsPCA | 1.143193 | 0.778 | +0.365193 |
| Unemployment | 12 | PLS | 1.230348 | 0.849 | +0.381348 |
| Unemployment | 24 | PCA | 1.023068 | 0.842 | +0.181068 |
| Unemployment | 24 | SPCA | 1.059398 | 0.800 | +0.259398 |
| Unemployment | 24 | sPCA | 1.026485 | 0.812 | +0.214485 |
| Unemployment | 24 | SsPCA | 1.017649 | 0.785 | +0.232649 |
| Unemployment | 24 | PLS | 1.091418 | 0.853 | +0.238418 |

The caveats are part of the result:

- The leak-free grid confounds two differences: the methodology leak and the data pipeline. The package run uses `load_fred_md` vintage `2023-04` and package transforms, not the exact author workbooks.
- The unemployment h=1 leak-free ratios, 3.5 to 4.9 versus paper values around 1.4 to 1.7, are 97.7% driven by the 2020 COVID point in the B2 diagnostic. The cell files show the mechanism directly: in May 2020, SsPCA predicts a +20.1085 unemployment change while the realized change is -1.5. That single row contributes 466.929 of 582.538 SsPCA h=1 SSE, or about 80.2%. The target file verifies that unemployment is built as the change in unemployment, not the level. This is not a package/data bug.
- The COVID result is itself a finding. The author's leaky standardization dampens COVID-period instability, while the leak-free forecast surface exposes it.
- The clean isolation of "difference equals leak" is the matched-data inflation comparison. There, leak-free package-side ratios differ from the paper by roughly 0.1 to 0.2: PCA 1.080953 versus 0.970, sPCA 0.964519 versus 0.768, and PLS 1.037423 versus 0.861. The `load_fred_md` grid also carries data-vintage and COVID effects.

## What the replication delivered

P1, trust: the package reproduces Table 2 on the author's methodology in the locked author-oracle grid, and it documents why the honest package output differs.

P2, bugs and findings: the major finding is the paper's look-ahead target-standardization leak. The B2 work also found and closed package or runner gaps, including a model-selection silent override, `parallel_cell_timeout`, and the missing method/config surfaces needed for this replication.

P3, efficiency: the K-prefix grouped evaluator made the supervised Table 2 run feasible, moving the workload from weeks to about 4.3 hours for the locked author-oracle run.

P4, statistical identity: the identity gate verified the K-prefix speedup before use, with `max_forecast_abs_diff = 0.0` and `max_score_abs_diff = 0.0`. The locked author-oracle grid did not use reduced grids, reduced folds, or reduced origins.

Package additions now on main include `pcr`, `ar_bic`, PLS raw-weight score projection, `standardize_scope`, `nan_policy`, `score_aggregation`, `preselect_stage`, and the K-prefix evaluator.

## Provenance / caveats

The author IP-growth and unemployment source exists in the reproducibility ZIP. A provenance check showed that using the target-specific IP-growth and unemployment source panels would close the five author-oracle misses in the locked 55/60 grid. Those misses are source-panel provenance limitations, not evidence of a `macroforecast` model defect.

The replication deliberately uses `macroforecast` to compute forecasts. The author's MATLAB serves only as a documented oracle for reading the methodology and proving the leak. No `macroforecast/**` package patch is part of this final documentation pass.
