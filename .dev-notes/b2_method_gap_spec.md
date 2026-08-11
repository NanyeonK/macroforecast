# B2 Method Gap Specification: Hounyo-Li Oracle vs Package Models

Scope: read-only method enumeration for the Hounyo-Li B2 fix lane. No `macroforecast/**`
changes and no model reruns were performed. The author oracle is the MATLAB code under
`qa/hounyo_li_matlab/Reproducibility package/Empirical/Macro/`.

## 1. Per-method Gap Table

| Arm | Current mapping | Author algorithm from MATLAB oracle | Current package behavior | Verdict |
|---|---|---|---|---|
| `AR_BIC` | `ar` | Per origin, `inflation_lag.m:78-96` and `inflation_arbic.m:82-97` take `y_or = y_loop(i:i+t_adj+h-1)`, apply `diff`, subtract `movmean(...,12)`, standardize the resulting target block, and form `wt = y_t`, `ytplush = y_{t+h}`. `lag_bic.m:8-35` searches lags 1..12 using `fitlm` with intercept and BIC `n*log(mean(resid^2)) + k*log(n)` where `k = lag * size(valid_data,2)`. `inflation_arbic.m:101-109` fits MATLAB `ar(insample_data,p)`, takes `phi = -A(2:end)`, and forecasts with `((phi).^h) * x'`. | `run_g1_smoke.py:127-137` maps to package `ar` with information-criterion search over `n_lag=1..12`. Package `ar` is fixed-order OLS or direct projection onto target lag columns (`macroforecast/models/timeseries.py:61-129`, `:153-180`), and package IC uses standard Gaussian IC from exposed `ssr_`, `nobs_`, `n_params_` (`macroforecast/model_selection/search.py:686-778`). It does not apply the author's `diff -> movmean(12)` target preparation, MATLAB `ar` backend, elementwise-power forecast, or the author's BIC parameter count. | **DIFFERENT / partly MISSING**. Add a general `ar_bic`/AR-backend capability if the benchmark is to be reusable. The author's target standardization leak is not a package addition. |
| `PCA` | `far` | `PCA_emp002.m:5-16` splits the last column as OOS, residualizes `ytplush_insample` on `wt=[target lag; constant]`, then `PCA_emp002.m:20-27` computes SVD of `xt_insample`, forms top-K PC scores, regresses the residual target on those scores, and forecasts `factor_part + control_part`. `PCA_PC2.m:20-28` adds a separate squared-factor head for PC2. | Package `far` is a factor-augmented autoregression/direct model: in direct mode it extracts PCA factors from the predictor block, appends target lags, and runs one joint regression (`macroforecast/models/timeseries.py:1080-1157`, `:1222-1247`). It is selected as `PCA` in `run_g1_smoke.py:139-146`. | **MISSING** as a reusable atomic model. Author PCA is residualized PCR, not `far`. Add general `pcr` with optional control residualization. |
| `sPCA` | `scaled_pca` | `inflation_linear.m:199-211` estimates one marginal predictive slope per standardized predictor using `ytplush` and scales each row of `xt_standardized`; `scaledPCA_emp002.m:5-16` then splits OOS and residualizes on controls; `scaledPCA_emp002.m:20-27` SVDs `scaleXs_insample`, regresses residual target on scaled-PC scores, and adds the control forecast. `scaledPCA_tune.m:17-22`, `:51-91`, `:96-104` tune K by three rolling folds. | Package `scaled_pca` computes marginal slopes, scaled-PCA factors, optional control residualization, and factor forecast head (`macroforecast/models/linear.py:1514-1634`; slopes/state at `:1578-1588`, `:1948-1979`). With `scale=False`, `control_columns`, `include_constant=True`, and pre-standardized inputs, the atomic estimator matches the leak-free version of the author method. | **MATCH for the reusable leak-free estimator; DIFFERENT for the exact author oracle** because the MATLAB script estimates scaling slopes after full-block target standardization that includes the realized OOS target. Do not add a leaky slope-scope option. |
| `SPCA` | `supervised_pca` | `SPCA_emp002.m:15-30` residualizes on controls and initializes `xt0`, `ytplush0`; `SPCA_emp002.m:32-60` iteratively selects the top `qN` predictors by absolute residual correlation, extracts one PC with `svds`, regresses residual target on that component, projects both y and X residuals, and repeats K times; `SPCA_emp002.m:67-69` forecasts `Gammahat*x_out + control`. `SPCA_tune.m:7-20` defines qN grid rules and `:22-27`, `:57-131`, `:136-149` tune K/qN. | Package `supervised_pca` implements the same iterative residual-correlation screening, SVD loading, alpha/lambda projection, and control forecast (`macroforecast/models/linear.py:1637-1790`, helper at `:2016-2094`). Current model preselection options run after model-internal scaling (`:1697-1733`), whereas author threshold scripts preselect raw `xt_non` before standardization (`inflation_linear.m:156-171`, `:189-197`). | **MATCH for unthresholded linear/SPCA estimator with `scale=False`, `preselect="none"`; DIFFERENT for thresholded variants** unless a raw-before-standardize preselect stage is added. Exact full-block target standardization remains a documented leak. |
| `SsPCA` | `supervised_scaled_pca` | `inflation_linear.m:199-211` first creates `scaleXs`; `SsPCA_emp002.m:15-30` residualizes on controls; `SsPCA_emp002.m:32-60` runs the same SPCA residual-correlation/SVD/projection recursion on scaled predictors; `SsPCA_emp002.m:68-70` forecasts. `SsPCA_tune.m:7-20`, `:22-27`, `:57-131`, `:136-149` tune K/qN. | Package `supervised_scaled_pca` is `SupervisedPCARegressor` with `slope_scale=True` (`macroforecast/models/linear.py:1793-1799`, `:2196-2252`) and otherwise uses the same SPCA helper. It computes slopes inside each leak-free model fit (`:1737-1743`) rather than using a full origin block that includes the realized OOS target. | **MATCH for the reusable leak-free estimator; DIFFERENT for the exact author oracle** due to full-block slope/target standardization leakage. Do not reproduce that leak as a normal package option. |
| `PLS` | `pls` | `PLS_emp002.m:5-16` splits OOS and residualizes on controls; `PLS_emp002.m:20-26` calls MATLAB `plsregress(xt_insample', ytplush0', K)`, takes `stats.W`, constructs scores manually as `B' * xt_insample`, re-estimates factor coefficients by OLS, and forecasts `(alphahat * B') * x_out + control`. `PLS_PC2.m:20-28` adds a separate squared-score head. | Package `pls` residualizes controls but delegates component extraction to sklearn `PLSRegression(scale=False)` and uses `model.transform(...)` scores (`macroforecast/models/linear.py:1320-1442`, public wrapper `:1445-1511`). This is not the same documented score construction as `stats.W' * raw xt_insample`; sklearn/MATLAB centering and weight normalization may differ. | **DIFFERENT**. Correct existing `pls` with a general raw-weight score projection / MATLAB-NIPALS-compatible option, or add a separate general PLS-weight-score model. |

## 2. General Package Add/Correct Specifications

### A. Add `pcr`: Principal Component Regression With Optional Control Residualization

Status: genuinely new package model. It replaces the B2 `PCA -> far` mapping.

Correctness oracle: `PCA_emp002.m:5-27`; PC2 extension `PCA_PC2.m:20-28`.

Public parameters:

- `n_components: int`
- `control_columns: Sequence[str] | None = None`
- `include_constant: bool = True`
- `drop_control_columns: bool = True`
- `standardize: bool = True` or package-consistent `scale: bool = True`
- `standardize_ddof: int = 1`
- `nan_policy: {"raise", "zero_after_standardize", "fill_zero"} = "raise"`
- `quadratic_factors: bool = False`
- `quadratic_mode: {"separate", "joint"} = "separate"`; default `separate` matches `PCA_PC2.m`
- `svd_solver`, `random_state` only if needed for existing deterministic PCA conventions.

Fit math for sample rows `t=1..T`:

1. Split X into factor block `X_f` and control block `W`. Add a constant column to `W` when requested. If `drop_control_columns=True`, controls are not included in `X_f`.
2. If `standardize=True`, compute column means and standard deviations on the fit rows of `X_f`, transform `X_f`, and store the state. Zero or invalid scales become 1. If `nan_policy="zero_after_standardize"`, replace non-finite standardized entries by 0 after standardization.
3. Estimate control coefficients `a_w = pinv(W) y`. If there are no controls, use an empty control block and residual `r=y`.
4. Residualize target only: `r = y - W a_w`. Do not residualize factor scores on controls.
5. Compute SVD of `X_f` (samples x features) and take feature loadings `V_K`. Equivalently to the MATLAB orientation, with author matrix `X_f' = U S V'`, use `V_K = U[:,1:K]`.
6. Scores: `F = X_f V_K`.
7. Linear head: `b_1 = pinv(F) r`.
8. If `quadratic_factors=True`, fit `b_2 = pinv(F^2) r` separately, not jointly with `F`.

Predict math:

`y_hat = W_new a_w + F_new b_1 + 1{quadratic}(F_new^2 b_2)`, where
`F_new = X_f,new V_K` after applying stored standardization.

Registration/search:

- Register `pcr` in `macroforecast/models/specs.py` with `n_components` search spaces matching `scaled_pca`/`pls`.
- B2 should tune `n_components=1..10` with the author fold/grid config, but the model itself must be general.

### B. Correct `pls`: Raw-weight PLS Score Projection Option

Status: correction/option to existing model, or a new general sibling if backward compatibility is cleaner.

Correctness oracle: `PLS_emp002.m:20-26`; PC2 extension `PLS_PC2.m:20-28`.

Add parameters:

- `score_projection: {"transform", "x_weights_raw"} = "transform"`; keep current behavior as default.
- `backend: {"sklearn", "matlab_nipals"} = "sklearn"` or a narrower `algorithm` parameter if exact MATLAB-compatible NIPALS is implemented internally.
- Existing `control_columns`, `include_constant`, `drop_control_columns`, `quadratic_factors`, `scale`, `max_iter`, `tol` remain.

`score_projection="x_weights_raw"` fit/predict contract:

1. Build controls and residual target exactly as in current `pls`: `a_w = pinv(W)y`, `r = y-Wa_w`.
2. Estimate PLS weights `B` with the selected backend on `(X_f, r)` for `n_components`.
3. Construct factor scores manually as `F = X_f B`, using the model-input scale of `X_f`, rather than `PLSRegression.transform`.
4. Fit `b_1 = pinv(F) r`; if `quadratic_factors=True`, fit `b_2 = pinv(F^2) r` separately.
5. Predict `W_new a_w + (X_f,new B)b_1 + 1{quadratic}((X_f,new B)^2 b_2)`.

This option is general: it is "PLS using raw x-weight scores with an external forecast head", not a paper-specific model.

### C. Add `ar_bic` or Extend `ar` With Internal BIC Selection and Backend/Forecast Options

Status: benchmark capability currently missing for author-faithful AR_BIC. It must be general and target-only; target transformations should live outside the model.

Correctness oracle: `lag_bic.m:8-35`, `inflation_arbic.m:99-113`.

Public parameters:

- `min_lag: int = 1`
- `max_lag: int = 12`
- `criterion: {"aic", "aicc", "bic"} = "bic"`
- `include_constant: bool = True` for lag-selection regressions
- `ic_parameter_count: {"standard", "lag_square"} = "standard"`; `lag_square` reproduces `k = lag * size(valid_data,2)` when explicitly requested.
- `estimator: {"ols", "yule_walker", "burg", "matlab_ar"} = "ols"`; exact `matlab_ar` can be an internal compatible implementation, not a MATLAB dependency.
- `forecast_mode: {"iterated", "direct_lag_projection", "coefficient_power"} = "iterated"`; `coefficient_power` reproduces the script's `((phi).^h) * x'` when explicitly requested.
- `horizon: int = 1`

Fit/predict contract:

1. Input `y` is already transformed and leak-free standardized by the caller.
2. For each candidate lag p, fit an AR lag regression on available training `y` only and compute the requested IC.
3. Choose the lowest-IC lag with deterministic first-min tie handling.
4. Fit the final AR backend at selected p.
5. Forecast according to `forecast_mode`.

Do not bundle `diff`, `movmean(12)`, or the author target z-score leak into the model. Those are data/target-preprocessing concerns.

### D. Existing Estimators to Keep, With Exact B2 Parameters

No new estimator is needed for the leak-free versions of:

- `scaled_pca`: use `scale=False`, `control_columns=(target_lag_column,)`, `include_constant=True`, `drop_control_columns=True`, `quadratic_factors=False` for linear PC, and tune `n_components`.
- `supervised_pca`: use `scale=False`, `control_columns=(target_lag_column,)`, `include_constant=True`, `drop_control_columns=True`, `preselect="none"` for unthresholded PC, `min_abs_corr=0`, tune `n_components` and `n_selected`.
- `supervised_scaled_pca`: same as `supervised_pca`, with slope scaling supplied by the model.

For thresholded variants outside the no-threshold B2 smoke, add a general `preselect_stage` option to `supervised_pca` and `supervised_scaled_pca`:

- `preselect_stage: {"after_standardize", "raw_before_standardize"} = "after_standardize"` for backward compatibility.
- `raw_before_standardize` computes hard t-stat or elastic-net screening on the unstandardized factor block, then standardizes only selected predictors. This matches `inflation_linear.m:156-171` followed by `:189-197` when used with leak-free targets.

### E. Supporting General Config Needed by B2

These are not estimator names, but the main fix lane needs them to avoid paper-wrapper hacks.

1. Predictor standardization scope:
   - Add a general preprocessing option such as `standardize_scope="origin_available_predictors"` or `include_current_predictor_rows=True`.
   - It may include predictor rows with positions `<= origin_pos`, including the current row used for prediction.
   - It must exclude rows with positions `> origin_pos` and must not include unavailable target realizations.
   - This is leak-free for predictors and addresses the current fit-window-only restriction documented by `_prepare_origin_panel` and the target-availability masks.

2. Post-standardization non-finite fill:
   - Add `standardize_nan_fill=0.0` or `nan_policy="zero_after_standardize"` so a standardization step can replace non-finite standardized predictor entries after scaling.
   - Oracle: `inflation_linear.m:189-191` and `inflation_linear_tune.m:187-189`.

3. Model-selection fold aggregation:
   - Add `score_aggregation="mean_split"` (current default) vs `score_aggregation="mean_fold"`.
   - Author tuning computes MSE inside each of three folds and then averages fold MSEs (`PCA_tune.m:90-101`, `SPCA_tune.m:129-143`), whereas package `evaluate_candidate` currently averages the score list across emitted splits (`macroforecast/model_selection/runner.py:26-48`).
   - The option is general for blocked CV with unequal fold sizes.

## 3. Standardization Verdict: Leak or Config?

### Predictor X standardization

Verdict: **leak-free at the actual forecast origin, but transductive over the current predictor row.**

Evidence:

- `inflation_linear.m:146-147` sets `x_or = x_loop(i:i+t_adj-1,:)` and `xt_non = x_or'`.
- `inflation_linear.m:153-154` defines `T` from `yt(:,1+h:end)`; with `t_adj=240`, T is 240 for h=1.
- `inflation_linear.m:189-191` standardizes `xt` by `mean(xt,2)` and `std(xt_bar,0,2)` over those T predictor columns, then zero-fills non-finite standardized entries.
- `PCA_emp002.m:5-6` confirms the last X column is the out-of-sample predictor and the preceding columns are in-sample.

The X standardization uses rows through the forecast origin's predictor row; it does not use `x_{T+h}` or any predictor row after the origin. This can be supported by a general, provenance-recorded predictor preprocessing option.

CV caveat: the MATLAB tune scripts pre-standardize the full 240-column origin block before inner validation refits. For early validation pseudo-origins, this lets later validation-block predictor rows affect scaling. That is a nested-CV transductive convention, not the actual final-origin X leak question.

### Target y standardization

Verdict: **genuine look-ahead leak. Do not reproduce as a normal package addition.**

Evidence:

- `inflation_linear.m:150-151` sets `y_or = y_loop(i:i+t_adj+h-1,:)` and `yt = y_or'`.
- `inflation_linear.m:192-194` computes `T1 = size(yt,2)` and standardizes all T1 target entries.
- `inflation_linear.m:196-197` sets controls `wt = yt_standardized(:,1:T)` and forecast targets `ytplush = yt_standardized(:,1+h:T+h)`.
- For h=1, T is 240 and T1 is 241, so `ytplush(:,end)` is standardized using the realized h-ahead target `y_{T+h}`. The method helpers then use that last `ytplush` column as `yt_outofsample` (`PCA_emp002.m:11-12`, `:27-28`; same pattern in the other helpers).
- AR_BIC has the same issue after its own `diff -> movmean(12)` target preparation: `inflation_arbic.m:82-97` standardizes the full target block before `outofsample_data = ytplush(:,end)` at `:101`.

Decision: add only the leak-free predictor-side standardization config. The target-side full-block z-score is a documented divergence from package-safe pseudo-OOS forecasting, not a method or config option to add.

## 4. Final B2 Arm Re-mapping

| B2 arm | Package model after fix lane | Status |
|---|---|---|
| `AR_BIC` benchmark | To-be-added `ar_bic` or extended `ar` with internal BIC/backend/forecast options | Existing `ar` is not exact. Use leak-free target preprocessing; document the author's target-standardization leak. |
| `PCA` | To-be-added `pcr` | Replace current `far` mapping. |
| `sPCA` | Existing `scaled_pca` | Keep, with `scale=False`, target-lag control, constant, and author K grid. Exact leaky full-block slope scaling is not reproduced. |
| `SPCA` | Existing `supervised_pca` | Keep for no-threshold linear PC with `scale=False`, `preselect="none"`, target-lag control, constant, K/qN grid. Add `preselect_stage` only for thresholded variants. |
| `SsPCA` | Existing `supervised_scaled_pca` | Keep for leak-free supervised scaled PCA with same controls/grid. Exact leaky full-block slope scaling is not reproduced. |
| `PLS` | Corrected `pls` with `score_projection="x_weights_raw"` / MATLAB-compatible weight option, or new general sibling | Current sklearn-transform score path is not the author oracle. |

## STOP Summary

Per-method verdicts:

| Method | Verdict |
|---|---|
| `AR_BIC` | DIFFERENT / partly MISSING |
| `PCA` | MISSING (`pcr`, residualized PCR) |
| `sPCA` | MATCH for leak-free estimator; DIFFERENT for exact leaky oracle |
| `SPCA` | MATCH for unthresholded estimator; DIFFERENT for thresholded preselect order |
| `SsPCA` | MATCH for leak-free estimator; DIFFERENT for exact leaky oracle |
| `PLS` | DIFFERENT |

Methods/params/config to add:

- Add general `pcr`.
- Correct/add PLS raw-weight score projection option.
- Add/extend `ar_bic` capability.
- Add predictor-only `origin_available` standardization scope.
- Add post-standardization non-finite fill.
- Add grouped fold-score aggregation for blocked CV.
- Add `preselect_stage="raw_before_standardize"` for thresholded SPCA/SsPCA variants.

Standardization decision:

- X standardization over the 240-row block is leak-free at the actual forecast origin because it uses rows `<= T`.
- Target standardization over the 241-row block uses realized `y_{T+h}` and is a genuine leak. Do not add a normal package option for it.
